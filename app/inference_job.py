"""Background video-inference job manager.

Mirrors the threading pattern already used by app/main.py for real-generator
batches: submit a video + model path, poll status via JSON, read annotated
artifacts when done.

One active job per process is enough for a classroom demo, but the store is
keyed by job_id so multiple concurrent jobs work correctly.

Outputs under outputs/inference/<job_id>/:
    annotated.mp4       — input video with drawn bboxes
    predictions.json    — COCO-style list of predictions
    summary.json        — config, wall time, per-class counts, optional metrics
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image

from generator.config import OUTPUT_DIR
from training.config import BBOX_COLORS


INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
INFERENCE_STATE_PATH = os.path.join(OUTPUT_DIR, "inference_state.json")


# ---------------------------------------------------------------------------
# Persistent model-path memory
# ---------------------------------------------------------------------------

def load_inference_state() -> Dict[str, Any]:
    if not os.path.exists(INFERENCE_STATE_PATH):
        return {}
    try:
        with open(INFERENCE_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_inference_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(INFERENCE_STATE_PATH), exist_ok=True)
    with open(INFERENCE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class InferenceJob:
    job_id: str
    video_path: str
    model_path: str
    threshold: float
    iou: float
    every_n: int
    max_frames: int
    gt_annotations_path: Optional[str] = None

    created_ts: float = field(default_factory=time.time)
    running: bool = True
    done: bool = False
    progress_frames: int = 0
    total_frames: int = 0
    wall_seconds: Optional[float] = None
    error: Optional[str] = None

    per_class_counts: Dict[str, int] = field(default_factory=dict)
    frames_with_detection: int = 0
    total_detections: int = 0
    avg_score: float = 0.0
    metrics: Optional[Dict[str, Any]] = None

    annotated_video: Optional[str] = None
    predictions_json: Optional[str] = None
    summary_json: Optional[str] = None

    # ----- serialization for the JSON status endpoint -----
    def snapshot(self) -> Dict[str, Any]:
        progress_pct = 0
        if self.total_frames > 0:
            progress_pct = int(round(100.0 * self.progress_frames / self.total_frames))
        progress_pct = max(0, min(100, progress_pct))
        return {
            "job_id": self.job_id,
            "running": self.running,
            "done": self.done,
            "progress_frames": self.progress_frames,
            "total_frames": self.total_frames,
            "progress_pct": progress_pct,
            "wall_seconds": self.wall_seconds,
            "error": self.error,
            "per_class_counts": dict(self.per_class_counts),
            "frames_with_detection": self.frames_with_detection,
            "total_detections": self.total_detections,
            "avg_score": round(float(self.avg_score), 4),
            "metrics": self.metrics,
            "model_path": self.model_path,
            "threshold": self.threshold,
            "iou": self.iou,
            "every_n": self.every_n,
            "max_frames": self.max_frames,
            "annotated_video_url": (
                f"/inference/annotated/{self.job_id}.mp4" if self.annotated_video else None
            ),
            "predictions_url": (
                f"/inference/predictions/{self.job_id}.json" if self.predictions_json else None
            ),
        }


_jobs: Dict[str, InferenceJob] = {}
_jobs_lock = threading.Lock()
_detector_cache: Dict[str, Any] = {}  # model_path -> YoloDetector


def get_job(job_id: str) -> Optional[InferenceJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> List[Dict[str, Any]]:
    with _jobs_lock:
        return [j.snapshot() for j in sorted(_jobs.values(), key=lambda j: -j.created_ts)]


# ---------------------------------------------------------------------------
# Detector cache (so back-to-back runs don't reload weights)
# ---------------------------------------------------------------------------

def _get_detector(model_path: str):
    model_path = str(Path(model_path).expanduser())
    cached = _detector_cache.get(model_path)
    if cached is not None:
        return cached
    from training.yolo_inference import YoloDetector
    detector = YoloDetector(model_path)
    _detector_cache[model_path] = detector
    return detector


def probe_detector(model_path: str) -> Dict[str, Any]:
    """Load and return info() — used by /inference/set-model to validate."""
    detector = _get_detector(model_path)
    return detector.info()


# ---------------------------------------------------------------------------
# Drawing (matches training.inference.draw_detections but pure-cv2 BGR)
# ---------------------------------------------------------------------------

def _draw_detections(frame_bgr, detections: List[Dict[str, Any]]):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det["label"]
        score = det["score"]
        color = BBOX_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame_bgr, text, (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return frame_bgr


# ---------------------------------------------------------------------------
# Metrics (recall / precision / FPR @ threshold)
# ---------------------------------------------------------------------------

def _iou_xywh(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0 else 0.0


def _compute_metrics(
    all_predictions: List[Dict[str, Any]],
    gt_by_frame_index: Dict[int, List[Dict[str, Any]]],
    label_to_gt_cat_id: Dict[str, int],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute frame-level recall / FPR plus detection-level precision.

    - Recall  = frames where we detected any pest class / frames that had >=1 GT pest
    - FPR     = negative frames (no GT) where we fired / total negative frames
    - Precision (detection-level) = true-positive dets / all dets, via greedy IoU match.
    """
    tp_dets = 0
    total_dets = 0
    pos_frames = 0
    hit_pos_frames = 0
    neg_frames = 0
    fp_neg_frames = 0

    for pred in all_predictions:
        frame_idx = pred["frame_index"]
        dets = pred.get("detections", [])
        total_dets += len(dets)
        gts = gt_by_frame_index.get(frame_idx, [])

        if gts:
            pos_frames += 1
            if dets:
                hit_pos_frames += 1
            # Greedy IoU match (class-aware)
            used_gt = set()
            for det in dets:
                det_cat_id = label_to_gt_cat_id.get(det["label"])
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gts):
                    if j in used_gt:
                        continue
                    if gt["category_id"] != det_cat_id:
                        continue
                    iou = _iou_xywh(det["bbox_xywh"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= iou_threshold:
                    tp_dets += 1
                    used_gt.add(best_j)
        else:
            neg_frames += 1
            if dets:
                fp_neg_frames += 1

    recall = (hit_pos_frames / pos_frames) if pos_frames else None
    fpr = (fp_neg_frames / neg_frames) if neg_frames else None
    precision = (tp_dets / total_dets) if total_dets else None

    return {
        "iou_threshold": iou_threshold,
        "pos_frames": pos_frames,
        "neg_frames": neg_frames,
        "frames_with_detection_on_positive": hit_pos_frames,
        "frames_with_detection_on_negative": fp_neg_frames,
        "true_detection_rate_frame": recall,
        "false_positive_rate_frame": fpr,
        "detection_precision": precision,
        "passes_target": (
            recall is not None and fpr is not None
            and recall >= 0.80 and fpr < 0.05
        ),
    }


def _load_gt_annotations(
    gt_path: str,
    sampled_frame_indices: List[int],
) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    """Parse a COCO annotations.json from the synthetic generator and return
    a map from *sampled* frame index (0..N-1 over frames we actually ran
    inference on) → list of {category_id, bbox:[x,y,w,h]} in image coords.

    Our generator writes one image entry per frame with file_name like
    frame_0001.png. We map frame_NNNN → index in sampled_frame_indices.
    """
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        return None

    # image_id -> file_name, then file_name -> zero-based frame index.
    id_to_name = {img["id"]: img["file_name"] for img in images}

    def _extract_frame_number(fname: str) -> Optional[int]:
        stem = os.path.splitext(os.path.basename(fname))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            return None
        try:
            # generator uses 1-based frame numbers
            return int(digits) - 1
        except ValueError:
            return None

    name_to_frame0 = {}
    for _id, name in id_to_name.items():
        f0 = _extract_frame_number(name)
        if f0 is not None:
            name_to_frame0[_id] = f0

    # Convert sampled_frame_indices list to a set for O(1) membership
    sampled_set = set(sampled_frame_indices)
    index_of_sample = {frame0: i for i, frame0 in enumerate(sampled_frame_indices)}

    by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        frame0 = name_to_frame0.get(ann.get("image_id"))
        if frame0 is None or frame0 not in sampled_set:
            continue
        sample_idx = index_of_sample[frame0]
        by_sample.setdefault(sample_idx, []).append({
            "category_id": int(ann.get("category_id", -1)),
            "bbox": list(ann.get("bbox", [0, 0, 0, 0])),
        })
    return by_sample


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def _run_inference(job: InferenceJob) -> None:
    t0 = time.time()
    try:
        detector = _get_detector(job.model_path)
    except Exception as e:
        job.error = f"Failed to load model: {e}"
        job.running = False
        job.done = True
        job.wall_seconds = round(time.time() - t0, 2)
        return

    cap = cv2.VideoCapture(job.video_path)
    if not cap.isOpened():
        job.error = f"Could not open video: {job.video_path}"
        job.running = False
        job.done = True
        job.wall_seconds = round(time.time() - t0, 2)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    every_n = max(1, int(job.every_n))
    max_frames = int(job.max_frames) if job.max_frames and job.max_frames > 0 else 0

    # Pre-count how many frames we'll sample so progress_pct is accurate.
    sampled_plan: List[int] = []
    for i in range(frame_count):
        if i % every_n == 0:
            sampled_plan.append(i)
            if max_frames and len(sampled_plan) >= max_frames:
                break
    job.total_frames = len(sampled_plan)

    out_dir = os.path.join(INFERENCE_DIR, job.job_id)
    os.makedirs(out_dir, exist_ok=True)
    annotated_path = os.path.join(out_dir, "annotated.mp4")
    predictions_path = os.path.join(out_dir, "predictions.json")
    summary_path = os.path.join(out_dir, "summary.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        job.error = "Could not open VideoWriter for annotated output"
        cap.release()
        job.running = False
        job.done = True
        job.wall_seconds = round(time.time() - t0, 2)
        return

    all_predictions: List[Dict[str, Any]] = []
    per_class: Dict[str, int] = {}
    frames_with_det = 0
    total_dets = 0
    score_sum = 0.0

    sampled_set = set(sampled_plan)
    frame_idx = 0
    sample_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx in sampled_set:
                # Ultralytics accepts BGR numpy or PIL; use PIL (RGB) for consistency.
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                try:
                    detections = detector.detect(pil, threshold=job.threshold, iou=job.iou)
                except Exception as e:
                    job.error = f"Detection failed on frame {frame_idx}: {e}"
                    detections = []

                if detections:
                    frames_with_det += 1
                for d in detections:
                    per_class[d["label"]] = per_class.get(d["label"], 0) + 1
                    total_dets += 1
                    score_sum += float(d["score"])

                all_predictions.append({
                    "frame_index": sample_idx,
                    "source_frame_index": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 3) if fps else None,
                    "detections": detections,
                })

                _draw_detections(frame_bgr, detections)
                job.progress_frames = sample_idx + 1
                sample_idx += 1

            writer.write(frame_bgr)
            frame_idx += 1

            if max_frames and sample_idx >= max_frames:
                # Finish writing the remaining unsampled frames so the output
                # stays the same length as the input would be. Drop them to
                # keep the video short; this matches "max_frames" intent.
                break
    finally:
        cap.release()
        writer.release()

    job.frames_with_detection = frames_with_det
    job.total_detections = total_dets
    job.per_class_counts = per_class
    job.avg_score = (score_sum / total_dets) if total_dets else 0.0

    # Optional GT metrics
    metrics = None
    if job.gt_annotations_path and os.path.exists(job.gt_annotations_path):
        gt_map = _load_gt_annotations(job.gt_annotations_path, sampled_plan)
        if gt_map is not None:
            from training.yolo_inference import yolo_label_to_coco_id
            label_to_gt = {
                label: cat_id
                for label in detector.class_names
                for cat_id in [yolo_label_to_coco_id(label)]
                if cat_id is not None
            }
            metrics = _compute_metrics(all_predictions, gt_map, label_to_gt, iou_threshold=0.5)
    job.metrics = metrics

    # ---- artifacts ----
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": os.path.basename(job.video_path),
            "model": os.path.basename(job.model_path),
            "threshold": job.threshold,
            "iou": job.iou,
            "every_n": every_n,
            "class_names": list(detector.class_names),
            "predictions": all_predictions,
        }, f, indent=2)

    job.wall_seconds = round(time.time() - t0, 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job.job_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": job.wall_seconds,
            "sampled_frames": job.total_frames,
            "source_frame_count": frame_count,
            "fps": fps,
            "per_class_counts": job.per_class_counts,
            "frames_with_detection": job.frames_with_detection,
            "total_detections": job.total_detections,
            "avg_score": round(float(job.avg_score), 4),
            "metrics": job.metrics,
            "threshold": job.threshold,
            "iou": job.iou,
            "every_n": every_n,
            "max_frames": max_frames,
            "model_path": job.model_path,
            "video_path": job.video_path,
            "gt_annotations_path": job.gt_annotations_path,
        }, f, indent=2)

    job.annotated_video = annotated_path
    job.predictions_json = predictions_path
    job.summary_json = summary_path
    job.running = False
    job.done = True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_inference_job(
    video_path: str,
    model_path: str,
    threshold: float = 0.5,
    iou: float = 0.45,
    every_n: int = 1,
    max_frames: int = 0,
    gt_annotations_path: Optional[str] = None,
) -> str:
    """Kick off an inference job in a background thread. Returns job_id."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    job_id = uuid.uuid4().hex[:10]
    job = InferenceJob(
        job_id=job_id,
        video_path=video_path,
        model_path=model_path,
        threshold=float(threshold),
        iou=float(iou),
        every_n=max(1, int(every_n)),
        max_frames=max(0, int(max_frames)),
        gt_annotations_path=gt_annotations_path,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(target=_run_inference, args=(job,), daemon=True).start()
    return job_id
