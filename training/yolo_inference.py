"""YOLOv8 inference helpers — framework-agnostic output shape.

The `detect()` function returns the SAME dict shape as training.inference.detect()
so downstream code (app/inference_job.py, visualization, metrics) works with both
backbones without branching.

Checkpoints are expected to be Ultralytics YOLOv8 weights (`*.pt`) trained with
class order matching `CLASS_NAMES` below. See PLAN.md Step 7.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from training.config import DETR_ID_TO_LABEL, get_device


# YOLO uses 0-indexed class ids. Our synthetic generator also writes labels in
# this order when exporting to YOLO format (see scripts/build_dataset.py), so
# index 0 == mouse, 1 == rat, 2 == cockroach.
CLASS_NAMES: List[str] = ["mouse", "rat", "cockroach"]


# ---------------------------------------------------------------------------
# Lazy import wrapper so the rest of the Flask app can be imported without
# ultralytics installed (e.g. on a machine that only runs the generator).
# ---------------------------------------------------------------------------

def _import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for YOLOv8 inference. "
            "Install with: pip install ultralytics"
        ) from e
    return YOLO


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YoloDetector:
    """Wraps an Ultralytics YOLO model with a stable output shape.

    Attributes
    ----------
    model_path : Path to the .pt checkpoint that was loaded.
    device     : torch device string ("cuda", "mps", "cpu").
    class_names: Ordered class names as reported by the checkpoint. Falls back
                 to CLASS_NAMES if the checkpoint omits them.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        YOLO = _import_yolo()
        self.model_path = str(Path(model_path).expanduser())
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")

        self.model = YOLO(self.model_path)

        dev = device or str(get_device())
        # Ultralytics accepts "cuda", "cpu", "mps", or an int. Normalize MPS:
        if dev.startswith("mps"):
            dev = "mps"
        self.device = dev

        # Prefer names embedded in the checkpoint; fall back to our canonical order.
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            ordered = [names[i] for i in sorted(names.keys())]
            self.class_names: List[str] = ordered or CLASS_NAMES
        elif isinstance(names, (list, tuple)) and len(names) > 0:
            self.class_names = list(names)
        else:
            self.class_names = list(CLASS_NAMES)

    # ---------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "class_names": list(self.class_names),
            "num_classes": len(self.class_names),
        }

    # ---------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        """Run detection on a single PIL image.

        Returns a list of dicts with keys:
            bbox_xyxy, bbox_xywh, score, label_id, label.
        Matches training.inference.detect() exactly, plus a 0-indexed label_id.
        """
        results = self.model.predict(
            source=image,
            conf=float(threshold),
            iou=float(iou),
            max_det=int(max_det),
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()

        detections: List[Dict[str, Any]] = []
        for box, score, cls_id in zip(xyxy, confs, cls_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            detections.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "score": round(float(score), 4),
                "label_id": int(cls_id),
                "label": label,
            })
        return detections


# ---------------------------------------------------------------------------
# COCO-style category id conversion
# ---------------------------------------------------------------------------

def yolo_label_to_coco_id(label: str) -> Optional[int]:
    """Return the COCO category id used by our synthetic generator.

    Our generator writes COCO labels as mouse=1, rat=2, cockroach=3 (see
    training/config.py::DETR_ID_TO_LABEL). YOLO stores them 0-indexed as
    mouse=0, rat=1, cockroach=2. When we compare predictions to ground truth
    we convert YOLO's 0-indexed label back to the COCO id via the label name.
    """
    inverse = {v: k for k, v in DETR_ID_TO_LABEL.items()}
    return inverse.get(label)
