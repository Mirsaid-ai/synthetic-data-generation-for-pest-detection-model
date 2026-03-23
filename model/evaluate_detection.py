"""
evaluate_detection.py
=====================
Evaluate DETR pest detection against ground truth COCO annotations.

Computes:
  - Standard COCO metrics: mAP@[0.5:0.95], mAP@0.5, AR
  - Project-specific metrics:
    - Detection Rate (Recall) at IoU >= 0.5  (target: >= 80%)
    - False Positive Rate at frame level      (target: < 5%)
    - Per-class breakdown

Usage:
    python evaluate_detection.py \
        --model_path ./detr_finetuned \
        --data_dir ./merged_dataset \
        --split test \
        --threshold 0.5
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_detection_on_dataset(model, processor, coco_gt, img_dir, device,
                              threshold=0.3):
    """Run model on all images in the COCO dataset, return predictions list."""
    predictions = []
    img_ids = coco_gt.getImgIds()

    for img_id in tqdm(img_ids, desc="Running detection"):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        # Handle relative paths -- file_name may be like "images/test/frame.jpg"
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            base_name = os.path.basename(file_name)
            img_path = os.path.join(img_dir, base_name)
        if not os.path.exists(img_path):
            continue

        pil_image = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": img_id,
                "category_id": label.item(),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": round(score.item(), 4),
            })

    return predictions


def compute_coco_metrics(coco_gt, predictions):
    """Run standard COCO evaluation."""
    if not predictions:
        print("[WARN] No predictions to evaluate")
        return {}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP_0.5_0.95": round(coco_eval.stats[0], 4),
        "mAP_0.5": round(coco_eval.stats[1], 4),
        "mAP_0.75": round(coco_eval.stats[2], 4),
        "AR_max1": round(coco_eval.stats[6], 4),
        "AR_max10": round(coco_eval.stats[7], 4),
        "AR_max100": round(coco_eval.stats[8], 4),
    }


def compute_project_metrics(coco_gt, predictions, iou_threshold=0.5):
    """
    Compute project-specific metrics:
    - Detection Rate (Recall): fraction of GT objects matched at IoU >= threshold
    - Frame-level FPR: fraction of pest-free frames where model falsely detects
    """
    img_ids = coco_gt.getImgIds()

    # Group predictions and GT by image
    gt_by_img = defaultdict(list)
    for img_id in img_ids:
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_by_img[img_id] = anns

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)

    tp_total = 0
    fn_total = 0
    fp_frames = 0
    tn_frames = 0
    per_class_tp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_fp = defaultdict(int)

    for img_id in img_ids:
        gt_anns = gt_by_img[img_id]
        preds = sorted(pred_by_img[img_id], key=lambda x: -x["score"])

        gt_boxes = np.array([a["bbox"] for a in gt_anns]) if gt_anns else np.zeros((0, 4))
        gt_cats = [a["category_id"] for a in gt_anns]
        gt_matched = [False] * len(gt_anns)

        has_gt = len(gt_anns) > 0
        has_pred = len(preds) > 0

        for pred in preds:
            px, py, pw, ph = pred["bbox"]
            pred_box = np.array([px, py, px + pw, py + ph])
            pred_cat = pred["category_id"]

            best_iou = 0.0
            best_idx = -1

            for gi, gt_ann in enumerate(gt_anns):
                if gt_matched[gi]:
                    continue
                gx, gy, gw, gh = gt_ann["bbox"]
                gt_box = np.array([gx, gy, gx + gw, gy + gh])

                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = pw * ph + gw * gh - inter
                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

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

        if not has_gt:
            if has_pred:
                fp_frames += 1
            else:
                tn_frames += 1

    detection_rate = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    fpr = fp_frames / (fp_frames + tn_frames) if (fp_frames + tn_frames) > 0 else 0.0

    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset.get("categories", [])}
    per_class = {}
    for cat_id in set(list(per_class_tp.keys()) + list(per_class_fn.keys())):
        name = cat_names.get(cat_id, str(cat_id))
        tp = per_class_tp[cat_id]
        fn = per_class_fn[cat_id]
        fp = per_class_fp[cat_id]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class[name] = {
            "tp": tp, "fn": fn, "fp": fp,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        }

    return {
        "detection_rate": round(detection_rate, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp_total,
        "fn": fn_total,
        "fp_frames": fp_frames,
        "tn_frames": tn_frames,
        "per_class": per_class,
        "detection_rate_pass": detection_rate >= 0.80,
        "fpr_pass": fpr < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR pest detection")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    data_root = Path(args.data_dir)
    ann_path = str(data_root / "annotations" / f"{args.split}.json")
    img_dir = str(data_root)

    if not os.path.exists(ann_path):
        print(f"[ERROR] Annotation file not found: {ann_path}")
        return

    print(f"Loading model from: {args.model_path}")
    processor = DetrImageProcessor.from_pretrained(args.model_path)
    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print(f"Loading annotations: {ann_path}")
    coco_gt = COCO(ann_path)
    print(f"Images: {len(coco_gt.getImgIds())} | Annotations: {len(coco_gt.getAnnIds())}")

    print(f"\nRunning detection (threshold={args.threshold})...")
    predictions = run_detection_on_dataset(
        model, processor, coco_gt, img_dir, device, threshold=args.threshold * 0.5
    )
    print(f"Total predictions: {len(predictions)}")

    print("\n" + "=" * 60)
    print("  COCO Standard Metrics")
    print("=" * 60)
    coco_metrics = compute_coco_metrics(coco_gt, predictions)
    for k, v in coco_metrics.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("  Project Metrics (threshold={:.2f})".format(args.threshold))
    print("=" * 60)

    high_conf_preds = [p for p in predictions if p["score"] >= args.threshold]
    project_metrics = compute_project_metrics(coco_gt, high_conf_preds)

    dr = project_metrics["detection_rate"]
    fpr = project_metrics["false_positive_rate"]
    dr_pass = "PASS" if project_metrics["detection_rate_pass"] else "FAIL"
    fpr_pass = "PASS" if project_metrics["fpr_pass"] else "FAIL"

    print(f"  Detection Rate (Recall): {dr:.1%}  [target >= 80%]  [{dr_pass}]")
    print(f"  False Positive Rate:     {fpr:.1%}  [target < 5%]   [{fpr_pass}]")
    print(f"  TP: {project_metrics['tp']}  FN: {project_metrics['fn']}")
    print(f"  FP frames: {project_metrics['fp_frames']}  TN frames: {project_metrics['tn_frames']}")

    print("\n  Per-class breakdown:")
    for name, stats in project_metrics["per_class"].items():
        print(f"    {name:12s}: recall={stats['recall']:.1%}  "
              f"precision={stats['precision']:.1%}  "
              f"(tp={stats['tp']}, fn={stats['fn']}, fp={stats['fp']})")

    if args.output_json:
        report = {
            "split": args.split,
            "threshold": args.threshold,
            "coco_metrics": coco_metrics,
            "project_metrics": project_metrics,
        }
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
