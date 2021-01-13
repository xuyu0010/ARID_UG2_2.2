import logging
import os
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

try:
	from . import initializer
except:
	import initializer


# Load pretrained into network
def load_state(network, state_dict):
	# customized partialy load function
	net_state_keys = list(network.state_dict().keys())
	net_state_keys_copy = net_state_keys.copy()
	sup_string = ""
	for key in state_dict.keys():
		if "backbone" in key:
			sup_string = "backbone."
		elif "module" in key:
			sup_string = "module."

	for i, _ in enumerate(net_state_keys_copy):
		name = net_state_keys_copy[i]
		if name.startswith('classifier') or name.startswith('fc'):
			continue

		if not sup_string:
			name_pretrained = name
		else:
			name_pretrained = sup_string + name

		if name_pretrained in state_dict.keys():
			dst_param_shape = network.state_dict()[name].shape
			if state_dict[name_pretrained].shape == dst_param_shape:
				network.state_dict()[name].copy_(state_dict[name_pretrained].view(dst_param_shape))
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
			logging.warning(">> Failed to load: {}".format(pruned_additional_states))

		return False

	return True

# Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None


# Gradient Scaling Layer
class GradScale(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output * ctx.beta
		return grad_input, None

