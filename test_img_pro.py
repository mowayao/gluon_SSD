from VOC_dataset.VOC_dataset import VOCDataset
from SSD.utils  import img_resize, vis_img, transformation
from SSD.config import Config as cfg

ds = VOCDataset(
        annotation_dir=cfg.annotation_dir,
        dataset_index=cfg.dataset_index,
        img_dir=cfg.img_dir, transform=transformation, resize_func=img_resize)

for i in [500, 105, 25, 40, 45, 100]:
	x, y = ds[i]
	vis_img(x, y, ds)