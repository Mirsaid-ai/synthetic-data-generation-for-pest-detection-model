"""Dataset class for DETR object detection."""

import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection

# ---------------------------------------------------------------------------
# DETR dataset — reads COCO-format annotations produced by the generator
# ---------------------------------------------------------------------------

def build_augmentation():
    """Augmentation transforms to reduce synthetic-to-real domain gap."""
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])


class CocoDetectionDETR(CocoDetection):
    """Wraps torchvision CocoDetection to return DETR-compatible batches.

    Expected dataset layout (produced by the generator pipeline):
        data_dir/
        ├── images/
        │   ├── train/  *.png
        │   ├── val/    *.png
        │   └── test/   *.png
        └── annotations/
            ├── train.json
            ├── val.json
            └── test.json
    """

    def __init__(self, img_folder, ann_file, processor, augment=False):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.augment_transform = build_augmentation() if augment else None

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)

        if self.augment_transform is not None:
            img = self.augment_transform(img)

        image_id = self.ids[idx]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        annotations = []
        for ann in ann_info:
            if ann.get("iscrowd", 0):
                continue
            annotations.append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": 0,
            })

        target_dict = {"image_id": image_id, "annotations": annotations}

        encoding = self.processor(
            images=img,
            annotations=[target_dict],
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return pixel_values, labels


def detr_collate_fn(batch):
    """Collate function for CocoDetectionDETR batches."""
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
