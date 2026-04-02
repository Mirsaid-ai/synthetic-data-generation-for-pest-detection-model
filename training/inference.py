"""Run DETR pest detection inference on images or video.

Usage:
    # Single image
    python -m training.inference --model_path ./detr_finetuned --image frame.jpg

    # Directory of images
    python -m training.inference --model_path ./detr_finetuned --image_dir ./test_frames/

    # Video file (extracts frames internally)
    python -m training.inference --model_path ./detr_finetuned --video test.mp4

    # Save COCO-format predictions
    python -m training.inference --model_path ./detr_finetuned --image_dir ./test/ \\
                                 --output_json predictions.json

    # Visualize detections
    python -m training.inference --model_path ./detr_finetuned --image_dir ./test/ \\
                                 --visualize --vis_dir ./vis_output/
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

from training.config import get_device, BBOX_COLORS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device):
    """Load a fine-tuned DETR model from a local directory."""
    print(f"Loading model from: {model_path}")
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(model, processor, image, device, threshold=0.5):
    """Run detection on a single PIL image.

    Returns:
        List of dicts with keys: bbox_xyxy, bbox_xywh, score, label_id, label.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.cpu().numpy().tolist()
        detections.append({
            "bbox_xyxy": box,
            "bbox_xywh": [box[0], box[1], box[2]-box[0], box[3]-box[1]],
            "score": round(score.item(), 4),
            "label_id": label.item(),
            "label": model.config.id2label.get(label.item(), f"class_{label.item()}"),
        })
    return detections


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections(image_bgr, detections):
    """Draw bounding boxes on an OpenCV BGR image in-place."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det["label"]
        score = det["score"]
        color = BBOX_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1-th-6), (x1+tw, y1), color, -1)
        cv2.putText(image_bgr, text, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image_bgr


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def collect_image_paths(args):
    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths.extend(sorted(
            p for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        ))
    return paths


def extract_video_frames(video_path, output_dir, every_n=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            fpath = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fpath, frame)
            paths.append(Path(fpath))
        idx += 1
    cap.release()
    print(f"Extracted {len(paths)} frames from {video_path}")
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DETR pest detection inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image",      default=None)
    parser.add_argument("--image_dir",  default=None)
    parser.add_argument("--video",      default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--vis_dir",    default="./vis_output")
    parser.add_argument("--every_n",    type=int, default=1,
                        help="For video: process every Nth frame")
    args = parser.parse_args()

    device = get_device()
    model, processor = load_model(args.model_path, device)

    image_paths = collect_image_paths(args)
    if args.video:
        tmp_dir = Path(args.vis_dir) / "_video_frames"
        image_paths.extend(extract_video_frames(args.video, str(tmp_dir), args.every_n))

    if not image_paths:
        print("No images found. Provide --image, --image_dir, or --video.")
        return

    print(f"Processing {len(image_paths)} images (threshold={args.threshold})")
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    all_predictions = []
    for img_path in tqdm(image_paths, desc="Detecting"):
        pil_image  = Image.open(img_path).convert("RGB")
        detections = detect(model, processor, pil_image, device, args.threshold)
        all_predictions.append({"file_name": str(img_path.name), "detections": detections})

        if detections:
            tqdm.write(f"  {img_path.name}: {len(detections)} — "
                       f"{[d['label'] for d in detections]}")

        if args.visualize:
            bgr = cv2.imread(str(img_path))
            vis = draw_detections(bgr, detections)
            cv2.imwrite(os.path.join(args.vis_dir, img_path.name), vis)

    total_dets   = sum(len(p["detections"]) for p in all_predictions)
    frames_with  = sum(1 for p in all_predictions if p["detections"])
    print(f"\nTotal: {total_dets} detections in {frames_with}/{len(all_predictions)} frames")

    if args.output_json:
        coco_preds = []
        for img_idx, pred in enumerate(all_predictions):
            for det in pred["detections"]:
                coco_preds.append({
                    "image_id": img_idx,
                    "category_id": det["label_id"],
                    "bbox": det["bbox_xywh"],
                    "score": det["score"],
                })
        with open(args.output_json, "w") as f:
            json.dump(coco_preds, f, indent=2)
        print(f"Predictions saved to: {args.output_json}")

    if args.visualize:
        print(f"Visualizations saved to: {args.vis_dir}")


if __name__ == "__main__":
    main()
