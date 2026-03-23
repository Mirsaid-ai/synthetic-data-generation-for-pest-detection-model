"""
finetune_detr.py
================
Fine-tune a DETR (DEtection TRansformer) model on a COCO-format pest dataset.

This replaces the ViT classification approach with proper object detection,
producing bounding boxes and class labels for each pest.

Usage:
    # Head-only training (freeze backbone)
    python finetune_detr.py --data_dir ../video_generator/pipeline_out/merged_dataset \
                            --freeze_backbone --epochs 20 --experiment_name exp1_head

    # Partial fine-tune (last 2 backbone stages)
    python finetune_detr.py --data_dir ../video_generator/pipeline_out/merged_dataset \
                            --partial_freeze 2 --epochs 30 --experiment_name exp2_partial

    # Full fine-tune
    python finetune_detr.py --data_dir ../video_generator/pipeline_out/merged_dataset \
                            --epochs 50 --experiment_name exp3_full

Dataset layout expected:
    data_dir/
    ├── images/
    │   ├── train/  *.jpg
    │   ├── val/    *.jpg
    │   └── test/   *.jpg
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CocoDetection
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

PEST_CATEGORIES = {1: "mouse", 2: "cockroach", 3: "rat"}
NUM_CLASSES = len(PEST_CATEGORIES)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_augmentation():
    """Data augmentation transforms to bridge synthetic-to-real domain gap."""
    return T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])


class CocoDetectionDETR(CocoDetection):
    """Wraps CocoDetection to return DETR-compatible format."""

    def __init__(self, img_folder, ann_file, processor, augment=False):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.augment_transform = build_augmentation() if augment else None

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self.augment_transform is not None:
            img = self.augment_transform(img)

        image_id = self.ids[idx]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        # Convert to DETR format: COCO uses [x,y,w,h], DETR needs same
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

        target_dict = {
            "image_id": image_id,
            "annotations": annotations,
        }

        encoding = self.processor(
            images=img,
            annotations=[target_dict],
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    pixel_values = torch.stack(pixel_values)
    return {"pixel_values": pixel_values, "labels": labels}


def apply_freeze_strategy(model, freeze_backbone, partial_freeze):
    if freeze_backbone:
        print("Strategy: HEAD-ONLY (backbone frozen)")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    elif partial_freeze > 0:
        print(f"Strategy: PARTIAL FINE-TUNE (unfreezing last {partial_freeze} backbone stages)")
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        backbone_layers = list(model.model.backbone.conv_encoder.model.layer_modules())
        if hasattr(model.model.backbone.conv_encoder.model, "layer4"):
            stages = ["layer1", "layer2", "layer3", "layer4"]
            unfreeze_stages = stages[-partial_freeze:]
            for stage_name in unfreeze_stages:
                stage = getattr(model.model.backbone.conv_encoder.model, stage_name)
                for param in stage.parameters():
                    param.requires_grad = True
    else:
        print("Strategy: FULL FINE-TUNE (all parameters trainable)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"[{split_name}]")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    return total_loss / max(n_batches, 1)


def save_results_to_csv(csv_path, row):
    fieldnames = [
        "experiment_name", "strategy", "epochs_trained",
        "best_val_loss", "final_train_loss", "lr",
    ]
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DETR for pest detection")
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to COCO dataset with images/ and annotations/ dirs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="./detr_finetuned")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--partial_freeze", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--results_csv", type=str, default="results_detection.csv")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (color jitter, blur, etc.)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        if args.freeze_backbone:
            args.lr = 1e-4
        elif args.partial_freeze > 0:
            args.lr = 5e-5
        else:
            args.lr = 1e-5

    device = get_device()
    print(f"Device: {device}")

    data_root = Path(args.data_dir)
    train_img_dir = str(data_root / "images" / "train")
    val_img_dir = str(data_root / "images" / "val")
    train_ann = str(data_root / "annotations" / "train.json")
    val_ann = str(data_root / "annotations" / "val.json")

    for p in [train_img_dir, val_img_dir, train_ann, val_ann]:
        if not os.path.exists(p):
            print(f"[ERROR] Not found: {p}")
            return

    print(f"Loading model: {args.model_name}")
    processor = DetrImageProcessor.from_pretrained(args.model_name)

    # DETR needs id2label mapping; we map our pest IDs
    id2label = {i: name for i, name in PEST_CATEGORIES.items()}
    label2id = {name: i for i, name in PEST_CATEGORIES.items()}

    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    apply_freeze_strategy(model, args.freeze_backbone, args.partial_freeze)
    model.to(device)

    print("Loading datasets...")
    train_dataset = CocoDetectionDETR(train_img_dir, train_ann, processor,
                                      augment=args.augment)
    val_dataset = CocoDetectionDETR(val_img_dir, val_ann, processor, augment=False)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": args.lr * 0.1},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    if args.freeze_backbone:
        strategy = "head_only"
    elif args.partial_freeze > 0:
        strategy = f"partial_{args.partial_freeze}stages"
    else:
        strategy = "full_finetune"

    best_val_loss = float("inf")
    print(f"\nStarting training: {args.experiment_name}")
    print(f"Strategy={strategy} | LR={args.lr} | Epochs={args.epochs}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device, "Val")
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}  "
              f"Val Loss={val_loss:.4f}  ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  -> New best val loss: {val_loss:.4f} -- model saved")

    print("-" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    save_results_to_csv(args.results_csv, {
        "experiment_name": args.experiment_name,
        "strategy": strategy,
        "epochs_trained": args.epochs,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": round(train_loss, 4),
        "lr": args.lr,
    })

    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
