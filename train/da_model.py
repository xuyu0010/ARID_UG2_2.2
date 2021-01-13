"""
Original Author: Yunpeng Chen
Adaptation Author: Yuecong Xu
"""
import os
import time
import socket
import logging
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metric
from . import callback

"""
Static Model
"""
class static_model(object):

	def __init__(self, net, criterion=None, criterion_domain=None, model_prefix='', DA_method=None, dom_rev=0.0, **kwargs):

		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		# init params
		self.net = net
		self.model_prefix = model_prefix
		self.criterion = criterion
		self.criterion_domain = criterion_domain
		self.DA_method = DA_method
		self.dom_rev = dom_rev

		logging.info("Domain Adaptation parameters: dom_rev={}".format(dom_rev))

		if self.DA_method is not None:
			logging.info("Using domain adaptation method: {}".format(DA_method))

	def load_state(self, state_dict, strict=False):
		if strict:
			self.net.load_state_dict(state_dict=state_dict)
		else:
			# customized partialy load function
			net_state_keys = list(self.net.state_dict().keys())
			for name, param in state_dict.items():
				if name in self.net.state_dict().keys():
					dst_param_shape = self.net.state_dict()[name].shape
					if param.shape == dst_param_shape:
						self.net.state_dict()[name].copy_(param.view(dst_param_shape))
						net_state_keys.remove(name)
			# indicating missed keys
			if net_state_keys:
				num_batches_list = []
				for i in range(len(net_state_keys)):
					if 'num_batches_tracked' in net_state_keys[i]:
						num_batches_list.append(net_state_keys[i])
				pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
				if pruned_additional_states:
					logging.info("There are layers in current network not initialized by pretrained")
					pruned = "[\'" + '\', \''.join(pruned_additional_states) + "\']"
					logging.warning(">> Failed to load: {}".format(pruned[0:150] + " ... " + pruned[-150:]))
				return False
		return True

	def get_checkpoint_path(self, epoch):
		assert self.model_prefix, "model_prefix undefined!"
		if torch.distributed.is_initialized():
			hostname = socket.gethostname()
			checkpoint_path = "{}_at-{}_ep-{:04d}.pth".format(self.model_prefix, hostname, epoch)
		else:
			checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
		return checkpoint_path

	def load_checkpoint(self, epoch, optimizer=None):

		load_path = self.get_checkpoint_path(epoch)
		assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

		checkpoint = torch.load(load_path)

		all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

		if optimizer:
			if 'optimizer' in checkpoint.keys() and all_params_matched:
				optimizer.load_state_dict(checkpoint['optimizer'])
				logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
			else:
				logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
		else:
			logging.info("Only model state resumed from: `{}'".format(load_path))

		if 'epoch' in checkpoint.keys():
			if checkpoint['epoch'] != epoch:
				logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

	def save_checkpoint(self, epoch, optimizer_state=None):

		save_path = self.get_checkpoint_path(epoch)
		save_folder = os.path.dirname(save_path)

		if not os.path.exists(save_folder):
			logging.debug("mkdir {}".format(save_folder))
			os.makedirs(save_folder)

		if not optimizer_state:
			torch.save({'epoch': epoch, 'state_dict': self.net.state_dict()}, save_path)
			logging.info("Checkpoint (only model) saved to: {}".format(save_path))
		else:
			torch.save({'epoch': epoch, 'state_dict': self.net.state_dict(), 'optimizer': optimizer_state}, save_path)
			logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


	def forward(self, src_data, tgt_data, src_label):
		
		if src_data is not None:
			src_data = src_data.float().cuda()
		else:
			src_data = torch.zeros(tgt_data.shape).float().cuda()
		if tgt_data is not None:
			tgt_data = tgt_data.float().cuda()
		if src_label is not None:
			src_label = src_label.cuda()
		if self.net.training:
			torch.set_grad_enabled(True)
		else:
			torch.set_grad_enabled(False)

		out_src, out_tgt, pred_dom_src, pred_dom_tgt = self.net(src_data, tgt_data, dom_rev=self.dom_rev)

		if hasattr(self, 'criterion') and self.criterion is not None and src_label is not None and self.net.training:
			output = out_src
			losses = []
			loss = self.criterion(out_src, src_label)
			losses.append('classification_loss: ' + str(loss.item()))
			if self.DA_method == 'DANN':
				loss_adv = 0
				for j in range(len(pred_dom_src)):
					src_domain_label = torch.zeros(pred_dom_src[j].size(0)).long()
					tgt_domain_label = torch.ones(pred_dom_tgt[j].size(0)).long()
					domain_label = torch.cat((src_domain_label,tgt_domain_label),0)
					if torch.cuda.is_available():
						domain_label = domain_label.cuda(non_blocking=True)
					domain_prediction = torch.cat((pred_dom_src[j], pred_dom_tgt[j]))
					loss_DANN = self.criterion_domain(domain_prediction, domain_label)
					loss_adv += loss_DANN
				loss += loss_adv
				losses.append('loss_adv: ' + str(loss_adv.item()))
		else:
			output = out_tgt
			loss = None
			losses = []
		return [output], [loss], losses


"""
Dynamic model that is able to update itself
"""
class da_model(static_model):

	def __init__(self, net, criterion, criterion_domain, model_prefix='', DA_method=None, dom_rev=0.0, 
				 step_callback=None, step_callback_freq=50, epoch_callback=None,
				 save_checkpoint_freq=1, opt_batch_size=None, **kwargs):

		# load parameters
		if kwargs:
			logging.warning("Unknown kwargs in model: {}".format(kwargs))

		super(da_model, self).__init__(net, criterion=criterion, criterion_domain=criterion_domain, model_prefix=model_prefix, DA_method=DA_method, dom_rev=dom_rev)

		# load optional arguments
		# - callbacks
		self.callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None, 'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None,}

		if not step_callback:
			step_callback = callback.CallbackList(callback.SpeedMonitor(), callback.MetricPrinter())

		if not epoch_callback:
			epoch_callback = (lambda **kwargs: None)

		self.step_callback = step_callback
		self.step_callback_freq = step_callback_freq
		self.epoch_callback = epoch_callback
		self.save_checkpoint_freq = save_checkpoint_freq
		self.batch_size=opt_batch_size


	"""
	In order to customize the callback function,
	you will have to overwrite the functions below
	"""
	def step_end_callback(self):
		# logging.debug("Step {} finished!".format(self.i_step))
		self.step_callback(**(self.callback_kwargs))

	def epoch_end_callback(self):
		self.epoch_callback(**(self.callback_kwargs))

		if self.callback_kwargs['epoch_elapse'] is not None:
			logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
					self.callback_kwargs['epoch'], self.callback_kwargs['epoch_elapse'], self.callback_kwargs['epoch_elapse']/3600.))

		if self.callback_kwargs['epoch'] == 0 or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
			self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1, optimizer_state=self.callback_kwargs['optimizer_dict'])

	"""
	Learning rate
	"""
	def adjust_learning_rate(self, lr, optimizer):
		for param_group in optimizer.param_groups:
			if 'lr_mult' in param_group:
				lr_mult = param_group['lr_mult']
			else:
				lr_mult = 1.0
			param_group['lr'] = lr * lr_mult

	"""
	Optimization
	"""
	def fit(self, src_train_iter, tgt_train_iter, optimizer, lr_scheduler, metrics=metric.Accuracy(topk=1), epoch_start=0, epoch_end=10000, **kwargs):

		"""
		checking
		"""
		if kwargs:
			logging.warning("Unknown kwargs: {}".format(kwargs))

		assert torch.cuda.is_available(), "only support GPU version"

		"""
		start the main loop
		"""
		pause_sec = 0.
		for i_epoch in range(epoch_start, epoch_end):
			self.callback_kwargs['epoch'] = i_epoch
			epoch_start_time = time.time()

			###########
			# 1] TRAINING
			###########
			metrics.reset()
			self.net.train()
			sum_sample_inst = 0
			sum_sample_elapse = 0.
			sum_update_elapse = 0
			batch_start_time = time.time()
			logging.info("Start epoch {:d}:".format(i_epoch))

			train_zip = zip(src_train_iter, cycle(tgt_train_iter)) if src_train_iter.__len__() > tgt_train_iter.__len__() else zip(src_train_iter, tgt_train_iter)
			train_iter = enumerate(train_zip)

			for i_batch, ((src_data, src_label), (tgt_data, _)) in train_iter:
				self.callback_kwargs['batch'] = i_batch

				update_start_time = time.time()

				# [forward] making next step
				if not tgt_data.shape[0] == src_data.shape[0]: continue
				outputs, losses, each_losses = self.forward(src_data, tgt_data, src_label)

				# [backward]
				optimizer.zero_grad()
				for loss in losses: loss.backward()
				self.adjust_learning_rate(optimizer=optimizer, lr=lr_scheduler.update())
				optimizer.step()

				# [evaluation] update train metric
				metrics.update([output.data.cpu() for output in outputs], src_label.cpu(), [loss.data.cpu() for loss in losses])

				# timing each batch
				sum_sample_elapse += time.time() - batch_start_time
				sum_update_elapse += time.time() - update_start_time
				batch_start_time = time.time()
				sum_sample_inst += src_data.shape[0]

				if (i_batch % self.step_callback_freq) == 0:
					# retrive eval results and reset metic
					self.callback_kwargs['namevals'] = metrics.get_name_value()
					metrics.reset()
					# speed monitor
					self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
					self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
					sum_update_elapse = 0
					sum_sample_elapse = 0
					sum_sample_inst = 0
					# callbacks
					self.step_end_callback()
					logging.info("The individual losses are: {}".format(each_losses))
				# torch.cuda.empty_cache()

			###########
			# 2] END OF EPOCH
			###########
			self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
			self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
			self.epoch_end_callback()

		logging.info("Optimization done!")
