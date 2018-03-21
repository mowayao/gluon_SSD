import mxnet.ndarray as nd


def flatten_preds(preds):
	return nd.flatten(nd.transpose(preds, axes=(0, 2, 3, 1)))

def concat_preds(preds):
	return nd.concat(*preds, dim=1)