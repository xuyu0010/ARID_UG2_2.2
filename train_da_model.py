import os
import logging

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim

from train import metric
from train.da_model import da_model
from data import da_iterator_factory as da_fac
from train.lr_scheduler import MultiFactorScheduler as MFS


def train_da_model(sym_net, model_prefix, DA_method, dom_rev, source_dataset, target_dataset, input_conf, clip_length=16, train_frame_interval=2, 
				resume_epoch=-1, batch_size=4, save_frequency=1, lr_base=0.01, lr_factor=0.1, lr_steps=[400000, 800000], end_epoch=1000, 
				distributed=False, fine_tune=False, **kwargs):

	assert torch.cuda.is_available(), "Currently, we only support CUDA version"

	# data iterator
	source_mean = target_mean = input_conf['mean']
	source_std = target_std = input_conf['std']
	iter_seed = torch.initial_seed() + (torch.distributed.get_rank() * 10 if distributed else 100) + max(0, resume_epoch) * 100
	src_train_iter = da_fac.creat(name=source_dataset, batch_size=batch_size, clip_length=clip_length, train_interval=train_frame_interval,
										mean=source_mean, std=source_std, seed=iter_seed)
	tgt_train_iter = da_fac.creat(name=target_dataset, batch_size=batch_size, clip_length=clip_length, train_interval=train_frame_interval,
										mean=target_mean, std=target_std, seed=iter_seed)
	
	# wapper (dynamic model)
	criterion = torch.nn.CrossEntropyLoss()
	criterion = criterion.cuda()
	criterion_domain = torch.nn.CrossEntropyLoss()
	criterion_domain = criterion_domain.cuda()
	net = da_model(net=sym_net, criterion=criterion, criterion_domain=criterion_domain, model_prefix=model_prefix, DA_method=DA_method, dom_rev=dom_rev, 
					step_callback_freq=50, save_checkpoint_freq=save_frequency, opt_batch_size=batch_size,)
	net.net.cuda()

	# config optimization
	param_base_layers = []
	param_new_layers = []
	param_freeze_layers = []
	name_base_layers = []
	for name, param in net.net.named_parameters():
		if fine_tune:
			# if name.startswith('classifier'):
			if 'classifier' in name or 'domain' in name or 'fc' in name:
				param_new_layers.append(param)
			elif 'conv1' in name or 'conv2' in name or 'maxpool' in name: # You may choose to freeze layers like this
				param.requires_grad = False
				param_freeze_layers.append(param)
			else:
				param_base_layers.append(param)
				name_base_layers.append(name)
		else:
			param_new_layers.append(param)

	if name_base_layers:
		out = "[\'" + '\', \''.join(name_base_layers) + "\']"
		logging.info("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers), out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))

	optims = optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2}, {'params': param_new_layers, 'lr_mult': 1.0}], lr=lr_base, momentum=0.9, weight_decay=0.0001,nesterov=True)

	# load params from pretrained 3d network
	if resume_epoch > 0:
		logging.info("Initializer:: resuming model from previous training")

	# resume training: model and optimizer
	if resume_epoch < 0:
		epoch_start = 0
		step_counter = 0
	else:
		net.load_checkpoint(epoch=resume_epoch, optimizer=optims)
		epoch_start = resume_epoch
		step_counter = epoch_start * train_iter.__len__()

	# set learning rate scheduler
	num_worker = dist.get_world_size() if torch.distributed.is_initialized() else 1
	lr_scheduler = MFS(base_lr=lr_base, steps=[int(x/(batch_size*num_worker)) for x in lr_steps], factor=lr_factor, step_counter=step_counter)
	
	# define evaluation metric
	metrics = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(name="top1", topk=1), metric.Accuracy(name="top5", topk=5),)

	net.fit(src_train_iter=src_train_iter, tgt_train_iter=tgt_train_iter, optimizer=optims, lr_scheduler=lr_scheduler, metrics=metrics, epoch_start=epoch_start, epoch_end=end_epoch)
