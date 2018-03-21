import mxnet as mx
from mxnet import autograd
from mxnet.contrib.ndarray import MultiBoxTarget, MultiBoxDetection

import mxnet.ndarray as F
from mxnet.gluon.data import DataLoader
from SSD.loss import FocalLoss, SmoothL1Loss, SoftmaxLoss
from SSD.utils import random_flip, img_norm, img_resize
from SSD.net import build_ssd
from VOC_dataset.VOC_dataset import  VOCDataset
from SSD.config import Config as cfg

import numpy as np
import time
from torchnet.logger import VisdomPlotLogger
import os

def train_transformation(data, label):
	data, label = random_flip(data, label)
	data = img_norm(data, cfg.mean, cfg.std)
	data = F.transpose(data, (2, 0, 1))
	return data, label

def training_targets(anchors, class_preds, label):
	class_preds = F.transpose(class_preds, axes=(0, 2, 1))
	z = MultiBoxTarget(anchors, label, class_preds, negative_mining_ratio=3, ignore_label=-1, overlap_threshold=0.5, negative_mining_thresh=0.5)
	box_target = z[0]
	box_mask = z[1]
	cls_target = z[2]
	#print (cls_target)
	return box_target, box_mask, cls_target

train_dataset = VOCDataset(annotation_dir=cfg.annotation_dir,
                           img_dir=cfg.img_dir,
                           dataset_index=cfg.dataset_index,
                           transform=train_transformation,
                           resize_func=img_resize)

'''
m = 0
n = np.inf
for _ in xrange(len(train_dataset)):
	img, gt = train_dataset[_]
	m = max(m, gt.shape[0])
	n = min(n, gt.shape[0])
print "max objects", m ,56
print "min objects", n ,1
'''
ctx = mx.gpu(3)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)




net = build_ssd("train", 300, ctx)
net.params_init(ctx)
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=60000, factor=0.5, stop_factor_lr=cfg.min_lr)
trainer = mx.gluon.trainer.Trainer(net.collect_params(),
                                    'sgd',
                                    {'learning_rate': cfg.base_lr,
                                     'wd': cfg.weight_decay,
                                     'momentum': 0.9,
									 'lr_scheduler': lr_scheduler})

#print train_dataloader
focal_cls_loss = FocalLoss(axis=1)
box_loss = SmoothL1Loss()
cls_loss = SoftmaxLoss()

if cfg.use_visdom:
	train_cls_loss_logger = VisdomPlotLogger(
		'line', port=7777, opts={'title': 'Train Classification Loss'}
	)
	train_reg_loss_logger = VisdomPlotLogger(
		'line', port=7777, opts={'title': 'Train Regression Loss'}
	)
cnt = 0
box_metric = mx.metric.MAE()
for epoch in xrange(1, cfg.epochs+1):
	tic = time.time()
	box_metric.reset()
	for iteration, (data, label) in enumerate(train_dataloader):
		btic = time.time()
		data = data.as_in_context(ctx)
		label = label.as_in_context(ctx)
		#true_label = label.copy()
		cls_loss_list = list()
		reg_loss_list = list()
		with mx.autograd.record():
			anchors, class_preds, box_preds = net(data)
			label[:, :, 1:] /= cfg.img_size

			label = F.concat(*[label, F.ones_like(label, ctx=ctx)*-1, F.ones_like(label, ctx=ctx)*-1], dim=1)
			box_target, box_mask, cls_target = training_targets(anchors, class_preds, label)
			loss1 = focal_cls_loss(class_preds[0], cls_target[0], -1)
			loss2 = box_loss(box_preds, box_target, box_mask)
			loss = loss1 + loss2
		cls_loss_list.append(F.mean(loss1)[0].asscalar())
		reg_loss_list.append(F.mean(loss2)[0].asscalar())
		loss.backward()
		trainer.step(cfg.batch_size)
		box_metric.update([box_target], [box_preds * box_mask])
		if (iteration + 1) % cfg.log_interval == 0:
			val1 = np.mean(cls_loss_list)
			val2 = np.mean(reg_loss_list)

			print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f, %s=%f' % (
				epoch, iteration, cfg.batch_size / (time.time() - btic), "cls loss", val1, "reg loss", val2, "box metric", box_metric.get()[1]))
			if cfg.use_visdom:
				train_cls_loss_logger.log(cnt, val1)
				train_reg_loss_logger.log(cnt, val2)
			cls_loss_list = []
			reg_loss_list = []
			cnt += 1
	print('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

if not os.path.exists("model"):
	os.mkdir("model")
net.save_params('model/ssd.params')
