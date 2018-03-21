import os
import cv2
import mxnet as mx
import numpy as np
from .parsexml import parseFile
from SSD.config import Config as cfg
class VOCDataset(mx.gluon.data.Dataset):
	voc_class_name = ['person', 'bird', 'cat', 'cow', 'dog',
					  'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
					  'bus', 'car', 'motorbike', 'train', 'bottle',
					  'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

	def __init__(self, annotation_dir, img_dir, dataset_index, transform = None, resize_func = None, **kwargs):
		super(VOCDataset, self).__init__(**kwargs)
		with open(dataset_index) as f:
			self.dataset_index = [t.strip() for t in f.readlines()]
		self.img_dir = img_dir
		self.annotation_dir = annotation_dir
		self.transform = transform
		self.class_to_id = {}
		self.resize_func = resize_func
		for i, class_name in enumerate(self.voc_class_name):
			self.class_to_id[class_name] = i
	def __getitem__(self, idx):
		idx = self.dataset_index[idx]
 		img_path = os.path.join(self.img_dir, idx + '.jpg')
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		anno_path = os.path.join(self.annotation_dir, idx + '.xml')
		standard_gt = np.ones(shape=(56, 5), dtype=np.float32) * -1.0
		gt = self.convert_gt_into_array(parseFile(anno_path))
		if self.resize_func is not None:
			img, scale = self.resize_func(img)
			scale = [scale*2]
		else:
			scale = 1
		gt[:, 1:] = gt[:, 1:].copy() * np.array(scale, dtype=np.float32)

		if self.transform != None:
			img, gt = self.transform(img, gt)
		gt[:, 1:] = gt[:, 1:] / cfg.img_size
		standard_gt[:len(gt)] = gt.copy()
		return img, standard_gt

	def __len__(self):
		return len(self.dataset_index)
	def convert_gt_into_array(self, gt, filter_difficult=True):
		"""
			Args:
				gt: the ground truth return by parseFile
				filter_difficult: filter out difficult cases or not
			Returns:
				A n * 5 array in the format of [x1, y1, x2, y2, c]
		"""
		ret = []
		for obj in gt['objects']:
			if filter_difficult and (obj['difficult'] == 1):
				pass
			new_array = list()
			new_array.append(self.class_to_id[obj['name']])
			new_array.extend(obj['bndbox'])
			ret.append(new_array)
		return np.asarray(ret, dtype=np.float32)