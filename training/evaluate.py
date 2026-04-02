"""Evaluate a fine-tuned DETR model against ground-truth COCO annotations.

Computes:
  - Standard COCO metrics: mAP@[0.5:0.95], mAP@0.5, AR
  - Project targets:
      Detection Rate (Recall) >= 80%
      False Positive Rate     <  5%
  - Per-class breakdown

Usage:
    python -m training.evaluate \\
        --model_path ./detr_finetuned \\
        --data_dir   outputs/dataset \\
        --split      test \\
        --threshold  0.5
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

from training.config import get_device, DETECTION_RATE_THRESHOLD, FPR_THRESHOLD, IOU_THRESHOLD


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_detection_on_dataset(model, processor, coco_gt, img_dir, device, threshold=0.3):
    """Run the model on all images in a COCO dataset. Returns a predictions list."""
    predictions = []
    img_ids = coco_gt.getImgIds()

    for img_id in tqdm(img_ids, desc="Running detection"):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, os.path.basename(file_name))
        if not os.path.exists(img_path):
            continue

        pil_image = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            predictions.append({
                "image_id": img_id,
                "category_id": label.item(),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": round(score.item(), 4),
            })

    return predictions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_coco_metrics(coco_gt, predictions):
    if not predictions:
        print("[WARN] No predictions to evaluate.")
        return {}
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP_0.5_0.95": round(coco_eval.stats[0], 4),
        "mAP_0.5":       round(coco_eval.stats[1], 4),
        "mAP_0.75":      round(coco_eval.stats[2], 4),
        "AR_max1":        round(coco_eval.stats[6], 4),
        "AR_max10":       round(coco_eval.stats[7], 4),
        "AR_max100":      round(coco_eval.stats[8], 4),
    }


def compute_project_metrics(coco_gt, predictions, iou_threshold=IOU_THRESHOLD):
    """Compute detection rate and frame-level false positive rate."""
    img_ids = coco_gt.getImgIds()
    gt_by_img   = defaultdict(list)
    pred_by_img = defaultdict(list)

    for img_id in img_ids:
        gt_by_img[img_id] = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)

    tp_total = fn_total = fp_frames = tn_frames = 0
    per_class_tp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_fp = defaultdict(int)

    for img_id in img_ids:
        gt_anns  = gt_by_img[img_id]
        preds    = sorted(pred_by_img[img_id], key=lambda x: -x["score"])
        gt_matched = [False] * len(gt_anns)
        gt_cats  = [a["category_id"] for a in gt_anns]

        for pred in preds:
            px, py, pw, ph = pred["bbox"]
            pred_cat = pred["category_id"]
            best_iou, best_idx = 0.0, -1

            for gi, gt_ann in enumerate(gt_anns):
                if gt_matched[gi]:
                    continue
                gx, gy, gw, gh = gt_ann["bbox"]
                x1 = max(px, gx); y1 = max(py, gy)
                x2 = min(px+pw, gx+gw); y2 = min(py+ph, gy+gh)
                inter = max(0, x2-x1) * max(0, y2-y1)
                union = pw*ph + gw*gh - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_idx = iou, gi

            if best_iou >= iou_threshold and best_idx >= 0:
                gt_matched[best_idx] = True
                tp_total += 1
                per_class_tp[pred_cat] += 1
            else:
                per_class_fp[pred_cat] += 1

        for gi, matched in enumerate(gt_matched):
            if not matched:
                fn_total += 1
                per_class_fn[gt_cats[gi]] += 1

        if not gt_anns:
            if preds:
                fp_frames += 1
            else:
                tn_frames += 1

    dr  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    fpr = fp_frames / (fp_frames + tn_frames) if (fp_frames + tn_frames) > 0 else 0.0

    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset.get("categories", [])}
    per_class = {}
    for cat_id in set(list(per_class_tp) + list(per_class_fn)):
        tp = per_class_tp[cat_id]; fn = per_class_fn[cat_id]; fp = per_class_fp[cat_id]
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class[cat_names.get(cat_id, str(cat_id))] = {
            "tp": tp, "fn": fn, "fp": fp,
            "recall": round(recall, 4), "precision": round(precision, 4),
        }

    return {
        "detection_rate": round(dr, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp_total, "fn": fn_total,
        "fp_frames": fp_frames, "tn_frames": tn_frames,
        "per_class": per_class,
        "detection_rate_pass": dr  >= DETECTION_RATE_THRESHOLD,
        "fpr_pass":            fpr <  FPR_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR pest detection")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ann_path = str(Path(args.data_dir) / "annotations" / f"{args.split}.json")
    if not os.path.exists(ann_path):
        print(f"[ERROR] Annotation file not found: {ann_path}")
        return

    processor = DetrImageProcessor.from_pretrained(args.model_path)
    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    coco_gt = COCO(ann_path)
    print(f"Images: {len(coco_gt.getImgIds())} | Annotations: {len(coco_gt.getAnnIds())}")

    img_dir = str(Path(args.data_dir) / "images" / args.split)
    predictions = run_detection_on_dataset(
        model, processor, coco_gt, img_dir, device,
        threshold=args.threshold * 0.5,
    )

    print("\n" + "=" * 60)
    print("  COCO Standard Metrics")
    print("=" * 60)
    coco_metrics = compute_coco_metrics(coco_gt, predictions)
    for k, v in coco_metrics.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print(f"  Project Metrics (threshold={args.threshold:.2f})")
    print("=" * 60)
    high_conf = [p for p in predictions if p["score"] >= args.threshold]
    pm = compute_project_metrics(coco_gt, high_conf)

    print(f"  Detection Rate: {pm['detection_rate']:.1%}  "
          f"[target >={DETECTION_RATE_THRESHOLD:.0%}]  "
          f"[{'PASS' if pm['detection_rate_pass'] else 'FAIL'}]")
    print(f"  False Positive Rate: {pm['false_positive_rate']:.1%}  "
          f"[target <{FPR_THRESHOLD:.0%}]  "
          f"[{'PASS' if pm['fpr_pass'] else 'FAIL'}]")
    print(f"  TP: {pm['tp']}  FN: {pm['fn']}  "
          f"FP frames: {pm['fp_frames']}  TN frames: {pm['tn_frames']}")

    print("\n  Per-class breakdown:")
    for name, stats in pm["per_class"].items():
        print(f"    {name:12s}: recall={stats['recall']:.1%}  "
              f"precision={stats['precision']:.1%}  "
              f"(tp={stats['tp']}, fn={stats['fn']}, fp={stats['fp']})")

    if args.output_json:
        report = {"split": args.split, "threshold": args.threshold,
                  "coco_metrics": coco_metrics, "project_metrics": pm}
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output_json}")


if __name__ == "__main__":
    main()
