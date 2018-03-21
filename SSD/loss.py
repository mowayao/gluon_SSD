import mxnet as mx
from mxnet import gluon as nn
import mxnet.ndarray as F

class SoftmaxLoss(nn.loss.Loss):
	def __init__(self, batch_axis=0, **kwargs):
		super(SoftmaxLoss, self).__init__(None, batch_axis, **kwargs)

	def hybrid_forward(self, F, x, y, ignore_label):
		output = F.log_softmax(x)
		label_matrix = mx.nd.zeros(output.shape, ctx=output.context)
		for i in xrange(label_matrix.shape[1]):
			label_matrix[:, i] = (y == i)
		ignore_unit = (y == ignore_label)
		loss = -F.sum(output * label_matrix, axis=1)
		return F.sum(loss) / (output.shape[0] - F.sum(ignore_unit))


class FocalLoss(nn.loss.Loss):
	def __init__(self, batch_axis=0, axis=2, alpha=0.25, gamma=2, **kwargs):
		super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
		self._axis = axis
		self._alpha = alpha
		self._gamma = gamma
	def hybrid_forward(self, F, x, y, ignore_label=-1.):
		output = F.softmax(x)
		pt = F.pick(output, y, axis=self._axis)
		mask = y != ignore_label
		loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt) * mask
		return F.sum(loss) / F.sum(mask)

class SmoothL1Loss(nn.loss.Loss):
	def __init__(self, batch_axis=0, **kwargs):
		super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
	def hybrid_forward(self, F, output, label, mask):
		loss = F.smooth_l1((output-label) * mask, scalar=1.0)
		return F.sum(loss) / F.sum(mask)