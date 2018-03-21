import numpy as np
import random
import cv2
from .config import Config as cfg
from mxnet import image as mx_img
import mxnet.ndarray as F
def random_flip(data, label):
	p = random.random()
	h, w, c = data.shape
	if p < 0.5:
		data = cv2.flip(data, 1)
		x1 = label[:, 1].copy()
		x3 = label[:, 3].copy()
		label[:, 1] = w - x3
		label[:, 3] = w - x1
	return data, label

def img_resize(img):
	h, w, c = img.shape
	img = cv2.resize(img, (cfg.img_size, cfg.img_size))
	return img, [cfg.img_size/float(w), cfg.img_size/float(h)]

def random_crop(img, label):
	#TODO:
	return img, label

def img_norm(img, mean, std):
	#img is a ndarray
	img = F.array(img) / 255.
	return mx_img.color_normalize(img, F.array(mean), F.array(std))

def transformation(data, label):

    data, label = random_flip(data, label)
    #data, label = random_square_crop(data, label)
    return data, label


def vis_img(img, label, ds):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	for item in label:
		cv2.rectangle(img, (int(item[1]), int(item[2])), (int(item[3]), int(item[4])), color=(255, 0, 0), thickness=2)
		#print item
		cv2.putText(img, ds.voc_class_name[int(item[0])], (int(item[1]), int(item[4])), 0, 0.5, (0, 255, 0))
	cv2.imshow("Img", img)
	cv2.waitKey(0)

