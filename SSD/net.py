import argparse
#import logging
#logging.basicConfig(level=logging.DEBUG)
from ops import flatten_preds, concat_preds
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxPrior
import mxnet.ndarray as F
from mxnet.gluon.model_zoo import vision
from .config import Config

v2 = Config.v2
scale = []
ratios = []
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4]
}


class L2Norm(nn.Block):
	def __init__(self, n_channels, scale):
		super(L2Norm, self).__init__()
		self.n_channels = n_channels
		self.eps = 1e-8
		self.weight = self.params.get(
			'weight', init=mx.init.Constant(scale),
			shape=(n_channels,))

	def forward(self, x):
		norm = (x ** 2).sum(axis=1, keepdims=True).sqrt() + self.eps
		x = F.divide(x, norm)
		out = self.weight.data().expand_dims(axis=0).expand_dims(axis=2).expand_dims(axis=3) * x
		return out

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = nn.HybridSequential()
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                with layers.name_scope():
					layers.add(nn.Conv2D(cfg[k + 1],
                           kernel_size=(1, 3)[flag], strides=2, padding=1))
            else:
                with layers.name_scope():
					layers.add(nn.Conv2D(v, kernel_size=(1, 3)[flag], padding=0))
            flag = not flag
        in_channels = v
    layers.hybridize()
    return layers



def vgg(cfg, batch_norm=False, ctx=None):
	vggnet = vision.vgg16(pretrained=True, ctx=ctx)
	pretrained_layers = list(vggnet.features)
	cur = 0
	layers = nn.HybridSequential()
	i = 0
	while i < len(cfg):
		v = cfg[i]
		if v == 'M':
			with layers.name_scope():
				layers.add(nn.MaxPool2D(pool_size=2, strides=2))
			i += 1
		elif v == 'C':
			with layers.name_scope():
				layers.add(nn.MaxPool2D(pool_size=2, strides=2, ceil_mode=True))
			i += 1
		elif isinstance(pretrained_layers[cur], nn.Conv2D):
			conv2d = pretrained_layers[cur]
			with layers.name_scope():
				layers.add(conv2d)
				layers.add(nn.Activation('relu'))
			i += 1
		cur += 1
	with layers.name_scope():
		layers.add(
			nn.MaxPool2D(pool_size=3, strides=1, padding=1),
			nn.Conv2D(1024, kernel_size=3, padding=6, dilation=6),
			nn.Activation('relu'),
			nn.Conv2D(1024, kernel_size=1),
			nn.Activation('relu')
		)
	layers.hybridize()
	return layers


class SSD(nn.Block):
	def __init__(self, phase, num_classes, cfg, **kwargs):
		super(SSD, self).__init__(**kwargs)
		self.anchor_size = None
		self.anchor_ratio = None
		self.num_classes = num_classes
		self.phase = phase
		self.num_classes = num_classes
		self.cfg = cfg
		self.size = 300

		# SSD network
		with self.name_scope():
			self.vgg = nn.HybridSequential()#base

		# Layer learns to scale the l2 normalized features from conv4_3
		with self.name_scope():
			self.L2Norm = L2Norm(512, 20)

		self.extras = nn.HybridSequential()
		self.loc = nn.HybridSequential()#head[0]
		self.conf = nn.HybridSequential()#head[1]
	def init_vgg(self, cfg, ctx):
		with self.name_scope():
			vggnet = vision.vgg16(pretrained=True, ctx=ctx)
		pretrained_layers = list(vggnet.features)
		cur = 0
		i = 0
		while i < len(cfg):
			v = cfg[i]
			if v == 'M':
				with self.name_scope():
					self.vgg.add(nn.MaxPool2D(pool_size=2, strides=2))
				i += 1
			elif v == 'C':
				with self.name_scope():
					self.vgg.add(nn.MaxPool2D(pool_size=2, strides=2, ceil_mode=True))
				i += 1
			elif isinstance(pretrained_layers[cur], nn.Conv2D):
				#conv2d = pretrained_layers[cur]
				with self.name_scope():
					self.vgg.add(nn.Conv2D(v, 3, padding=1))
					self.vgg.add(nn.Activation('relu'))
				i += 1
			cur += 1
		with self.vgg.name_scope():
			self.vgg.add(
				nn.MaxPool2D(pool_size=3, strides=1, padding=1),
				nn.Conv2D(1024, kernel_size=3, padding=6, dilation=6),
				nn.Activation('relu'),
				nn.Conv2D(1024, kernel_size=1),
				nn.Activation('relu')
			)
		self.vgg.hybridize()
	def init_extras(self, cfg, i):
		# Extra layers added to VGG for feature scaling

		in_channels = i
		flag = False
		for k, v in enumerate(cfg):
			if in_channels != 'S':
				if v == 'S':
					with self.name_scope():
						self.extras.add(nn.Conv2D(cfg[k + 1],
											 kernel_size=(1, 3)[flag], strides=2, padding=1))
				else:
					with self.name_scope():
						self.extras.add(nn.Conv2D(v, kernel_size=(1, 3)[flag], padding=0))
				flag = not flag
			in_channels = v
		self.extras.hybridize()
	def init_conf_box(self, cfg, num_classes):
		vgg_src = [21, -2]
		for k, v in enumerate(vgg_src):
			with self.name_scope():
				self.loc.add(nn.Conv2D(cfg[k] * 4, kernel_size=3, padding=1))
				self.conf.add(nn.Conv2D(cfg[k] * num_classes, kernel_size=3, padding=1))
		for k, v in enumerate(self.extras[1::2], 2):
			with self.name_scope():
				self.loc.add(nn.Conv2D(cfg[k] * 4, kernel_size=3, padding=1))
				self.conf.add(nn.Conv2D(cfg[k] * num_classes, kernel_size=3, padding=1))
		self.loc.hybridize()
		self.conf.hybridize()
	def params_init(self, ctx):
		self.collect_params().initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
		self.init_vgg_params()
	def init_vgg_params(self):
		for l1, l2 in zip(self.vgg, vision.vgg16(pretrained=True).features[:-4]):
			if isinstance(l1, nn.Conv2D):
				l1.weight.set_data(l2.weight.data())
				l1.bias.set_data(l2.bias.data())
	def forward(self, x):
		sources = list()
		loc = list()
		conf = list()
		priors = list()
		# apply vgg up to conv4_3 relu
		for k in range(23):
			x = self.vgg[k](x)

		s = self.L2Norm(x)
		sources.append(s)
		# apply vgg up to fc7
		for k in range(23, len(self.vgg)):
			x = self.vgg[k](x)
		sources.append(x)

		# apply extra layers and cache source layer outputs
		for k, v in enumerate(self.extras):
			x = F.relu(v(x))
			if k % 2 == 1:
				sources.append(x)
		for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
			boxes = MultiBoxPrior(x, sizes=self.cfg['sizes'][i], ratios=self.cfg['aspect_ratios'][i], clip=True)
			priors.append(boxes)
			l_res = l(x)
			c_res = c(x)
			loc.append(flatten_preds(l_res))
			conf.append(flatten_preds(c_res))

		priors = F.concat(*priors, dim=1)
		loc = F.concat(*loc, dim=1)
		conf = F.concat(*conf, dim=1)
		conf = F.reshape(conf, shape=(0, -1, self.num_classes))
		output = (priors, conf, loc)

		return output
def multibox(vgg, extras, cfg, num_classes):
	loc_net = nn.HybridSequential()
	conf_net = nn.HybridSequential()
	vgg_src = [21, -2]
	for k, v in enumerate(vgg_src):
		with loc_net.name_scope():
			loc_net.add(nn.Conv2D(cfg[k] * 4, kernel_size=3, padding=1))
		with conf_net.name_scope():
			conf_net.add(nn.Conv2D(cfg[k] * num_classes, kernel_size=3, padding=1))
	for k, v in enumerate(extras[1::2], 2):
		with loc_net.name_scope():
			loc_net.add(nn.Conv2D(cfg[k]* 4, kernel_size=3, padding=1))
		with conf_net.name_scope():
			conf_net.add(nn.Conv2D(cfg[k] * num_classes, kernel_size=3, padding=1))
	#
	conf_net.hybridize()
	loc_net.hybridize()
	return vgg, extras, (loc_net, conf_net)

def build_ssd(phase, size, ctx, num_classes=21):
	if phase != "test" and phase != "train":
		print("Error: Phase not recognized")
		return
	if size != 300:
		print("Error: Sorry only SSD300 is supported currently!")
		return

	#base_, extras_, head_ = multibox(vgg(base[str(size)], False, ctx),
	                                 #add_extras(extras[str(size)], 1024),
	                                 #mbox[str(size)], num_classes)
	net = SSD(phase, num_classes, v2)
	net.init_vgg(base[str(size)], ctx)
	net.init_extras(extras[str(size)], 1024)
	net.init_conf_box(mbox[str(size)], num_classes)
	return net
	#print (base_, extras_, head_)
	#return SSD(phase, base_, extras_, head_, num_classes, v2)
