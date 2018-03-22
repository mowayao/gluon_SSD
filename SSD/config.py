

class Config:
	#annotation_dir = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/Annotations'
	#dataset_index = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/ImageSets/Main/trainval.txt'
	#img_dir = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/JPEGImages'

	#test_annotation_dir = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/Annotations/'
	#test_dataset_index = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/ImageSets/Main/test.txt'
	#test_img_dir = '/media/mowayao/data/object_detection/VOC_dataset/VOC2012/JPEGImages/'
	annotation_dir = '../data/VOC_dataset/VOC2012/Annotations'
	dataset_index = '../data/VOC_dataset/VOC2012/ImageSets/Main/trainval.txt'
	img_dir = '../data/VOC_dataset/VOC2012/JPEGImages'

	test_annotation_dir = '../data/VOC_dataset/VOC2012/Annotations/'
	test_dataset_index = '../data/VOC_dataset/VOC2012/ImageSets/Main/test.txt'
	test_img_dir = '../data/object_detection/VOC_dataset/VOC2012/JPEGImages/'
	num_classes = 21
	batch_size = 1
	epochs = 100
	log_interval = 20
	decay_step = 2000 * 5
	decay_ratio = 0.5
	weight_decay = 5e-4
	seed = 1024
	use_visdom = True
	visdom_port = 7777
	base_lr = 1e-3
	min_lr = 1e-6
	nms_thresh = 0.3
	score_thresh = 0.5
	img_size = 300

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	num_workers = 2

	v2 = {
		'feature_maps': [38, 19, 10, 5, 3, 1],
		'min_dim': 300,
		'steps': [8, 16, 32, 64, 100, 300],
		'min_sizes': [30, 60, 111, 162, 213, 264],
		'max_sizes': [60, 111, 162, 213, 264, 315],
		'aspect_ratios': [[0.5, 2, 1], [2, 3, 1, 0.5, 1./3], [2, 3, 1, 0.5, 1./3], [2, 3, 1, 0.5, 1./3], [0.5, 2, 1], [0.5, 2, 1]],
		#'sizes': [[0.1, 0.2], [0.2, 0.333], [0.333, 0.55], [0.54, 0.71], [0.71, 0.88], [0.88, 1.05]],
		'sizes': [[0.1, 0.2], [0.2, 0.227], [0.333, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
		'variance': [0.1, 0.2],
		'clip': True,
		'name': 'v2',
	}

	# use average pooling layer as last layer before multibox layers
	v1 = {
		'feature_maps': [38, 19, 10, 5, 3, 1],
		'min_dim': 300,
		'steps': [8, 16, 32, 64, 100, 300],
		'min_sizes': [30, 60, 114, 168, 222, 276],
		'max_sizes': [-1, 114, 168, 222, 276, 330],
		# 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
		'aspect_ratios': [[1, 1, 2, 1 / 2], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3],
						  [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3]],
		'variance': [0.1, 0.2],
		'clip': True,
		'name': 'v1',
	}
