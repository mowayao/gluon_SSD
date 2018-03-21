from SSD.loss import FocalLoss, SoftmaxLoss
import mxnet.ndarray as F
import numpy as np
loss = FocalLoss(alpha=0.25, gamma=2, axis=1)
loss_soft = SoftmaxLoss()

x = F.array(np.ones(shape=(5, 21), dtype=np.float32))
y = F.array([[1, 1, 2, 3, -1]]).reshape((5, 1))
print x.shape, y.reshape((5, )).shape
print loss_soft(x, y.reshape((5, )), -1)





