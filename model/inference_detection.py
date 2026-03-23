"""
inference_detection.py
======================
Run DETR pest detection on images or video frames.

Usage:
    # Single image
    python inference_detection.py --model_path ./detr_finetuned --image frame.jpg

    # Directory of images
    python inference_detection.py --model_path ./detr_finetuned --image_dir ./test_frames/

    # Video file (extracts frames internally)
    python inference_detection.py --model_path ./detr_finetuned --video test.mp4

    # Save COCO-format predictions JSON
    python inference_detection.py --model_path ./detr_finetuned --image_dir ./test/ \
                                  --output_json predictions.json

    # Visualize detections
    python inference_detection.py --model_path ./detr_finetuned --image_dir ./test/ \
                                  --visualize --vis_dir ./vis_output/
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm

COLORS = {
    "mouse": (0, 200, 0),
    "cockroach": (0, 100, 255),
    "rat": (255, 100, 0),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def detect(model, processor, image, device, threshold=0.5):
    """Run detection on a single PIL image. Returns list of dicts with
    bbox (xyxy), score, label."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = box.cpu().numpy().tolist()
        detections.append({
            "bbox_xyxy": box,
            "bbox_xywh": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
            "score": round(score.item(), 4),
            "label_id": label.item(),
            "label": model.config.id2label.get(label.item(), f"class_{label.item()}"),
        })

    return detections


def draw_detections(image_bgr, detections):
    """Draw bounding boxes on an OpenCV BGR image."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det["label"]
        score = det["score"]
        color = COLORS.get(label, (200, 200, 200))

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(image_bgr, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image_bgr


def collect_image_paths(args):
    """Collect image paths from --image, --image_dir, or --video."""
    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        img_dir = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths.extend(sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts))
    return paths


def extract_video_frames(video_path, output_dir, every_n=1):
    """Extract frames from video, return list of frame paths."""
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


def main():
    parser = argparse.ArgumentParser(description="DETR pest detection inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="./vis_output")
    parser.add_argument("--every_n", type=int, default=1,
                        help="For video: process every Nth frame")
    args = parser.parse_args()

    device = get_device()
    model, processor = load_model(args.model_path, device)

    image_paths = collect_image_paths(args)

    if args.video:
        tmp_dir = Path(args.vis_dir) / "_video_frames"
        video_frames = extract_video_frames(args.video, str(tmp_dir), args.every_n)
        image_paths.extend(video_frames)

    if not image_paths:
        print("No images found. Provide --image, --image_dir, or --video.")
        return

    print(f"Processing {len(image_paths)} images (threshold={args.threshold})")

    all_predictions = []

    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Detecting"):
        pil_image = Image.open(img_path).convert("RGB")
        detections = detect(model, processor, pil_image, device, args.threshold)

        img_pred = {
            "file_name": str(img_path.name),
            "detections": detections,
        }
        all_predictions.append(img_pred)

        n_dets = len(detections)
        if n_dets > 0:
            labels = [d["label"] for d in detections]
            tqdm.write(f"  {img_path.name}: {n_dets} detection(s) - {labels}")

        if args.visualize:
            bgr = cv2.imread(str(img_path))
            vis = draw_detections(bgr, detections)
            vis_path = os.path.join(args.vis_dir, img_path.name)
            cv2.imwrite(vis_path, vis)

    # Summary
    total_dets = sum(len(p["detections"]) for p in all_predictions)
    frames_with_dets = sum(1 for p in all_predictions if p["detections"])
    print(f"\nTotal detections: {total_dets} across {frames_with_dets}/{len(all_predictions)} frames")

    if args.output_json:
        # Convert to COCO predictions format
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
