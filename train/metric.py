"""
Metric function library
Author: Yunpeng Chen
* most of the code are inspired by MXNet
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvalMetric(object):

	def __init__(self, name, **kwargs):
		self.name = str(name)
		self.reset()

	def update(self, preds, labels, losses):
		raise NotImplementedError()

	def reset(self):
		self.num_inst = 0
		self.sum_metric = 0.0

	def get(self):
		if self.num_inst == 0:
			return (self.name, float('nan'))
		else:
			return (self.name, self.sum_metric / self.num_inst)

	def get_name_value(self):
		name, value = self.get()
		if not isinstance(name, list):
			name = [name]
		if not isinstance(value, list):
			value = [value]
		return list(zip(name, value))

	def check_label_shapes(self, preds, labels):
		# raise if the shape is inconsistent
		if (type(labels) is list) and (type(preds) is list):
			label_shape, pred_shape = len(labels), len(preds)
		else:
			label_shape, pred_shape = labels.shape[0], preds.shape[0]

		if label_shape != pred_shape:
			raise NotImplementedError("")


class MetricList(EvalMetric):
	"""Handle multiple evaluation metric
	"""
	def __init__(self, *args, name="metric_list"):
		assert all([issubclass(type(x), EvalMetric) for x in args]), \
			"MetricList input is illegal: {}".format(args)
		self.metrics = [metric for metric in args]
		super(MetricList, self).__init__(name=name)

	def update(self, preds, labels, losses=None):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels
		losses = [losses] if type(losses) is not list else losses

		for metric in self.metrics:
			metric.update(preds, labels, losses)

	def reset(self):
		if hasattr(self, 'metrics'):
			for metric in self.metrics:
				metric.reset()
		else:
			logging.warning("No metric defined.")

	def get(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get())
		return ouputs

	def get_name_value(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get_name_value())        
		return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
	"""Computes accuracy classification score.
	"""
	def __init__(self, name='accuracy', topk=1):
		super(Accuracy, self).__init__(name)
		self.topk = topk

	def update(self, preds, labels, losses):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels

		self.check_label_shapes(preds, labels)
		for pred, label in zip(preds, labels):
			assert self.topk <= pred.shape[1], \
				"topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
			_, pred_topk = pred.topk(self.topk, 1, True, True)

			pred_topk = pred_topk.t()
			correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))

			self.sum_metric += float(correct.view(-1).float().sum(0, keepdim=True).numpy())
			self.num_inst += label.shape[0]


class Loss(EvalMetric):
	"""Dummy metric for directly printing loss.
	"""        
	def __init__(self, name='loss'):
		super(Loss, self).__init__(name)

	def update(self, preds, labels, losses):
		assert losses is not None, "Loss undefined."
		for loss in losses:
			self.sum_metric += float(loss.numpy().sum())
			self.num_inst += loss.numpy().size


# discrepancy loss used in MCD (CVPR 18)
def dis_MCD(out1, out2):
	return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2, dim=1)))


# correlation norm related losses
def hard_norm_dis(corr, norm_type='nuc', dim=(1,2), norm_tgt=25, norm_dis_weight=0.05, dis_type='l2'):
	if type(dim) is tuple:
		norm = torch.norm(corr, p=norm_type, dim=dim)
	else:
		norm = torch.norm(corr, p=norm_type)	
	if dis_type == 'l2':
		dis_loss = (norm.mean() - norm_tgt) ** 2 # L2 distance with norm target
	elif dis_type == 'smoothl1':
		norm_tgt = norm_tgt + torch.zeros(norm.shape[0])
		if torch.cuda.is_available():
			norm_tgt = norm_tgt.cuda()
		loss = nn.SmoothL1Loss()
		dis_loss = loss(norm, norm_tgt)
	return norm_dis_weight * dis_loss


def soft_norm_dis(corr, norm_type='nuc', dim=(1,2), norm_dis_weight=0.05):
	if type(dim) is tuple:
		norm_tgt = torch.norm(corr, p=norm_type, dim=dim).detach()
		norm = torch.norm(corr, p=norm_type, dim=dim)
	else:
		norm_tgt = torch.norm(corr, p=norm_type).detach()
		norm = torch.norm(corr, p=norm_type)
	assert norm_tgt.requires_grad == False
	norm_tgt = norm_tgt + 2.0
	dis_loss = ((norm - norm_tgt) ** 2).mean()
	return norm_dis_weight * dis_loss # L2 distance with norm target


def bnm_loss(tgt_feat, bnm_loss_weight=1.0):
	softmax_tgt = nn.Softmax(dim=1)(tgt_feat)
	loss_BNM = -torch.norm(softmax_tgt,'nuc')
	return bnm_loss_weight * loss_BNM


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	n_samples = int(source.size()[0])+int(target.size()[0])
	total = torch.cat([source, target], dim=0)
	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # For feature MMD, until size(1)
	# total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
	# total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))) # For corr MMD, until size(2)
	L2_distance = ((total0-total1)**2).sum(2) # For feature MMD, use only a single sum(2)
	# L2_distance = ((total0-total1)**2).sum(2).sum(2) # For corr MMD, use two sum(2)
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
	bandwidth /= kernel_mul ** (kernel_num // 2)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
	return sum(kernel_val)


def CORAL(source, target):
	d = source.data.shape[1]
	# source covariance
	xm = torch.mean(source, 0, keepdim=True) - source
	xc = xm.t() @ xm
	# target covariance
	xmt = torch.mean(target, 0, keepdim=True) - target
	xct = xmt.t() @ xmt
	# frobenius norm between source and target
	loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
	loss = loss/(4*d*d)
	return loss


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
	batch_size = int(source.size()[0])
	kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

	loss = 0

	if ver==1:
		for i in range(batch_size):
			s1, s2 = i, (i + 1) % batch_size
			t1, t2 = s1 + batch_size, s2 + batch_size
			loss += kernels[s1, s2] + kernels[t1, t2]
			loss -= kernels[s1, t2] + kernels[s2, t1]
		loss = loss.abs_() / float(batch_size)
	elif ver==2:
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY - YX)
	else:
		raise ValueError('ver == 1 or 2')

	return loss


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	batch_size = int(source.size()[0])
	kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

	loss1 = 0
	for s1 in range(batch_size):
		for s2 in range(s1+1, batch_size):
			t1, t2 = s1+batch_size, s2+batch_size
			loss1 += kernels[s1, s2] + kernels[t1, t2]
	loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

	loss2 = 0
	for s1 in range(batch_size):
		for s2 in range(batch_size):
			t1, t2 = s1+batch_size, s2+batch_size
			loss2 -= kernels[s1, t2] + kernels[s2, t1]
	loss2 = loss2 / float(batch_size * batch_size)
	return loss1 + loss2


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[2, 5], fix_sigma_list=[None, None], ver=2):
	batch_size = int(source_list[0].size()[0])
	layer_num = len(source_list)
	joint_kernels = None
	for i in range(layer_num):
		source = source_list[i]
		target = target_list[i]
		kernel_mul = kernel_muls[i]
		kernel_num = kernel_nums[i]
		fix_sigma = fix_sigma_list[i]
		kernels = guassian_kernel(source, target,
								  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
		if joint_kernels is not None:
			joint_kernels = joint_kernels * kernels
		else:
			joint_kernels = kernels

	loss = 0

	if ver==1:
		for i in range(batch_size):
			s1, s2 = i, (i + 1) % batch_size
			t1, t2 = s1 + batch_size, s2 + batch_size
			loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
			loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
		loss = loss.abs_() / float(batch_size)
	elif ver==2:
		XX = joint_kernels[:batch_size, :batch_size]
		YY = joint_kernels[batch_size:, batch_size:]
		XY = joint_kernels[:batch_size, batch_size:]
		YX = joint_kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY - YX)
	else:
		raise ValueError('ver == 1 or 2')

	return loss


if __name__ == "__main__":
	import torch

	# Test Accuracy
	predicts = [torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]]))]
	labels   = [torch.from_numpy(np.array([   0,            1,          1 ]))]
	losses   = [torch.from_numpy(np.array([   0.3,       0.4,       0.5   ]))]

	logging.getLogger().setLevel(logging.DEBUG)
	logging.debug("input pred:  {}".format(predicts))
	logging.debug("input label: {}".format(labels))
	logging.debug("input loss: {}".format(labels))

	acc = Accuracy()

	acc.update(preds=predicts, labels=labels, losses=losses)

	logging.info(acc.get())

	# Test MetricList
	metrics = MetricList(Loss(name="ce-loss"),
						 Accuracy(topk=1, name="acc-top1"), 
						 Accuracy(topk=2, name="acc-top2"), 
						 )
	metrics.update(preds=predicts, labels=labels, losses=losses)

	logging.info("------------")
	logging.info(metrics.get())
	acc.get_name_value()