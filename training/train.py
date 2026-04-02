"""Training entry point for DETR object detection.

Usage (local):
    python -m training.train --data_dir outputs/dataset --epochs 20 --freeze_backbone

Usage (Hugging Face):
    python -m training.train --hf_dataset your-username/pest-detection-dataset --freeze_backbone
    # Downloads the dataset from HF Hub into a local cache, then trains normally.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import (
    get_device,
    DETR_BATCH_SIZE, DETR_NUM_EPOCHS, DETR_WEIGHT_DECAY, DETR_MODEL_NAME,
)
from training.dataset import CocoDetectionDETR, detr_collate_fn
from training.model import create_detr_model, apply_freeze_strategy


# ---------------------------------------------------------------------------
# DETR training helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        pv = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pv, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
    return total_loss / max(n, 1)


@torch.no_grad()
def _evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"[{split_name}]")
    for batch in pbar:
        pv = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pv, labels=labels)
        total_loss += outputs.loss.item()
        n += 1
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})
    return total_loss / max(n, 1)


def _save_results_csv(csv_path, row):
    fieldnames = ["experiment_name", "strategy", "epochs_trained",
                  "best_val_loss", "final_train_loss", "lr"]
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train pest detection model")
    parser.add_argument("--model_name", default=DETR_MODEL_NAME)
    parser.add_argument("--data_dir", default=None,
                        help="Path to COCO dataset with images/ and annotations/ dirs")
    parser.add_argument("--hf_dataset", default=None,
                        help="Hugging Face dataset repo (e.g. username/pest-detection-dataset). "
                             "Downloads to a local cache and sets --data_dir automatically.")
    parser.add_argument("--batch_size", type=int, default=DETR_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DETR_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", default="./detr_finetuned")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--partial_freeze", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--experiment_name", default="experiment")
    parser.add_argument("--results_csv", default="results_detection.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hf_dataset is not None:
        from huggingface_hub import snapshot_download
        print(f"Downloading dataset from HF Hub: {args.hf_dataset}")
        args.data_dir = snapshot_download(repo_id=args.hf_dataset, repo_type="dataset")
        print(f"Dataset cached at: {args.data_dir}")

    if args.data_dir is None:
        parser.error("--data_dir or --hf_dataset is required for DETR training")

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
    train_img = str(data_root / "images" / "train")
    val_img   = str(data_root / "images" / "val")
    train_ann = str(data_root / "annotations" / "train.json")
    val_ann   = str(data_root / "annotations" / "val.json")

    for p in [train_img, val_img, train_ann, val_ann]:
        if not os.path.exists(p):
            print(f"[ERROR] Not found: {p}")
            return

    print(f"Loading model: {args.model_name}")
    model, processor = create_detr_model(args.model_name)
    apply_freeze_strategy(model, args.freeze_backbone, args.partial_freeze)
    model.to(device)

    train_ds = CocoDetectionDETR(train_img, train_ann, processor, augment=args.augment)
    val_ds   = CocoDetectionDETR(val_img,   val_ann,   processor, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=detr_collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=detr_collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],     "lr": args.lr * 0.1},
    ], weight_decay=DETR_WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    strategy = ("head_only" if args.freeze_backbone
                else f"partial_{args.partial_freeze}stages" if args.partial_freeze
                else "full_finetune")

    best_val_loss = float("inf")
    train_loss = 0.0
    print(f"\nStarting DETR training: {args.experiment_name}")
    print(f"Strategy={strategy} | LR={args.lr} | Epochs={args.epochs}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = _evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:3d}: Train={train_loss:.4f}  Val={val_loss:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  -> Best val loss {val_loss:.4f} — model saved to {args.output_dir}")

    print("-" * 60)
    print(f"Done. Best val loss: {best_val_loss:.4f}")

    _save_results_csv(args.results_csv, {
        "experiment_name": args.experiment_name,
        "strategy": strategy,
        "epochs_trained": args.epochs,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": round(train_loss, 4),
        "lr": args.lr,
    })


if __name__ == "__main__":
    main()
