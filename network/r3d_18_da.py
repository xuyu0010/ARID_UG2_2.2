"""
Author: Yuecong Xu
"""
import logging
import os

import torch
import torch.nn as nn

try:
	from .util import load_state
	from .util import GradReverse as gradrev
except:
	from util import load_state
	from util import GradReverse as gradrev


class Conv3DSimple(nn.Conv3d):
	def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):

		super(Conv3DSimple, self).__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)

	@staticmethod
	def get_downsample_stride(stride):
		return (stride, stride, stride)


class BasicBlock(nn.Module):

	expansion = 1

	def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
		midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

		super(BasicBlock, self).__init__()
		self.conv1 = nn.Sequential(
			conv_builder(inplanes, planes, midplanes, stride),
			nn.BatchNorm3d(planes),
			nn.ReLU(inplace=True)
		)
		self.conv2 = nn.Sequential(
			conv_builder(planes, planes, midplanes),
			nn.BatchNorm3d(planes)
		)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.conv2(out)
		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class BasicStem(nn.Sequential):
	"""The default conv-batchnorm-relu stem
	"""
	def __init__(self):
		super(BasicStem, self).__init__(
			nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
					  padding=(1, 3, 3), bias=False),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

	def __init__(self, block, conv_makers, layers, stem, num_classes=400, zero_init_residual=False, DA_method=None):
		"""Generic resnet video generator.
		Args:
			block (nn.Module): resnet building block
			conv_makers (list(functions)): generator function for each layer
			layers (List[int]): number of blocks per layer
			stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
			num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
			zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
		"""
		super(VideoResNet, self).__init__()
		self.inplanes = 64
		self.ReLU = nn.ReLU(inplace=True)

		self.stem = stem()

		self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.DA_method = DA_method

		if self.DA_method == 'DANN':
			logging.info("Network:: Using DANN based Domain Adaptation with base DA network")
			self.dom_feat_fc = nn.Linear(512, 512)
			self.dom_class_fc = nn.Linear(512, 2)

		# init weights
		self._initialize_weights()

		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)

	def forward(self, src_x, tgt_x, dom_rev=1.0):
		src_x = self.stem(src_x)
		tgt_x = self.stem(tgt_x)

		src_x = self.layer1(src_x)
		tgt_x = self.layer1(tgt_x)
		src_x = self.layer2(src_x)
		tgt_x = self.layer2(tgt_x)
		src_x = self.layer3(src_x)
		tgt_x = self.layer3(tgt_x)
		src_x = self.layer4(src_x)
		tgt_x = self.layer4(tgt_x)

		src_x = self.avgpool(src_x)
		tgt_x = self.avgpool(tgt_x)
		src_x = src_x.flatten(1)
		tgt_x = src_x.flatten(1)

		if self.DA_method == 'DANN':
			dom_src = []
			dom_tgt = []

			# DANN on src_h
			dom_src_x = gradrev.apply(src_x, dom_rev)
			dom_src_x = self.dom_class_fc(self.ReLU(self.dom_feat_fc(dom_src_x)))
			dom_tgt_x = gradrev.apply(tgt_x, dom_rev)
			dom_tgt_x = self.dom_class_fc(self.ReLU(self.dom_feat_fc(dom_tgt_x)))
			dom_src.append(dom_src_x)
			dom_tgt.append(dom_tgt_x)
		else:
			dom_src, dom_tgt = [], []

		src_x = self.fc(src_x)
		tgt_x = self.fc(tgt_x)

		return src_x, tgt_x, dom_src, dom_tgt

	def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
		downsample = None

		if stride != 1 or self.inplanes != planes * block.expansion:
			ds_stride = conv_builder.get_downsample_stride(stride)
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
				nn.BatchNorm3d(planes * block.expansion)
			)
		layers = []
		layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, conv_builder))

		return nn.Sequential(*layers)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, num_classes=400, DA_method=None, **kwargs):

	model = VideoResNet(num_classes=num_classes, DA_method=None, **kwargs)

	if pretrained: # Download pretrained model for this demo as in https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
		pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r3d_18-b3b3357e.pth')
		logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
		assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
		pretrained = torch.load(pretrained_model)
		load_state(model, pretrained)
	else:
		logging.info("Network:: graph initialized, use random inilization!")

	return model


def r3d_18_da(pretrained=False, num_classes=400, **kwargs):

	return _video_resnet('r3d_18', pretrained, block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2,2,2,2], stem=BasicStem, num_classes=num_classes, **kwargs)

if __name__ == "__main__":

	logging.getLogger().setLevel(logging.DEBUG)

	net = r3d_18_da(pretrained=True, num_classes=5, DA_method='DANN')
	src_data = torch.randn(1,3,16,224,224)
	tgt_data = torch.randn(1,3,16,224,224)
	if torch.cuda.is_available():
		net = net.cuda()
		src_data = src_data.cuda()
		tgt_data = tgt_data.cuda()
	src_out, tgt_out, dom_src, dom_tgt = net(src_data, tgt_data)
	print (src_out.shape)
	print (tgt_out.shape)
