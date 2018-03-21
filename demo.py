import sys
import os
import cv2
from SSD.config import Config as cfg
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as F
import random
from mxnet.ndarray.contrib import MultiBoxDetection
from SSD.net import build_ssd
from VOC_dataset.VOC_dataset import VOCDataset
def detect_image(img_path):
	if not os.path.exists(img_path):
		print('can not find image: ', img_path)
	# img = Image.open(img_file)
	#print img_path
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (cfg.img_size, cfg.img_size))
	# img = ImageOps.fit(img, [data_shape, data_shape], Image.ANTIALIAS)
	origin_img = img.copy()
	img = (img/255. - cfg.mean) / cfg.std
	img = np.transpose(img, (2, 0, 1))
	img = img[np.newaxis, :]
	img = F.array(img)

	print('input image shape: ', img.shape)

	ctx = mx.gpu(0)
	net = build_ssd("test", 300, ctx)
	net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
	net.collect_params().reset_ctx(ctx)
	params = 'model/ssd.params'
	net.load_params(params, ctx=ctx)

	anchors, cls_preds, box_preds = net(img.as_in_context(ctx))
	print('anchors', anchors)
	print('class predictions', cls_preds)
	print('box delta predictions', box_preds)
	# convert predictions to probabilities using softmax
	cls_probs = F.SoftmaxActivation(F.transpose(cls_preds, (0, 2, 1)), mode='channel')

	# apply shifts to anchors boxes, non-maximum-suppression, etc...
	output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress = True, clip = True, nms_threshold = 0.01)
	output = output.asnumpy()

	pens = dict()

	plt.imshow(origin_img)

	thresh = 0.3
	for det in output[0]:
		cid = int(det[0])
		if cid < 0:
			continue
		score = det[1]
		if score < thresh:
			continue
		if cid not in pens:
			pens[cid] = (random.random(), random.random(), random.random())
		scales = [origin_img.shape[1], origin_img.shape[0]] * 2
		xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
		rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=pens[cid], linewidth=3)
		plt.gca().add_patch(rect)
		voc_class_name = ['person', 'bird', 'cat', 'cow', 'dog',
						  'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
						  'bus', 'car', 'motorbike', 'train', 'bottle',
						  'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
		text = voc_class_name[cid]
		plt.gca().text(xmin, ymin - 2, '{:s} {:.3f}'.format(text, score),
					   bbox=dict(facecolor=pens[cid], alpha=0.5),
					   fontsize=12, color='white')
	plt.axis('off')
	# plt.savefig('result.png', dpi=100)
	plt.show()

if __name__ == '__main__':
	if len(sys.argv[1]) != 0:
		img_path =sys.argv[1]
		try:
			detect_image(img_path)
		except Exception as e:
			print(e)
			print('for detect please provide image file path.')
