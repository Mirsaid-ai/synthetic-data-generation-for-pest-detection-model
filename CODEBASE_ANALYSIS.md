# Codebase Analysis: Synthetic Pest Video Generator

> **Generated:** March 2026  
> **Repo:** `synthetic_video_gen`

---

## 1. What Is This Project?

This project generates **fully synthetic, annotated video datasets** for training pest-detection models (object detectors and instance segmenters). Instead of collecting and hand-labeling real footage of mice and cockroaches in kitchens, you provide a single kitchen photo and the pipeline:

1. Understands the 3-D geometry of the scene (depth + floor segmentation)
2. Procedurally animates pest sprites that walk only on the floor, scale with perspective, and cast contact shadows
3. Records pixel-perfect COCO bounding-box + segmentation annotations for every frame automatically
4. Packages everything as a ready-to-train COCO dataset split into `train / val / test`

The output is a standard COCO JSON dataset usable directly with Detectron2, MMDetection, YOLOv8, or any other COCO-compatible trainer.

---

## 2. File Map

| File | Role | Lines |
|------|------|-------|
| `run_pipeline.py` | End-to-end orchestrator; calls all other scripts in order | 307 |
| `generate_depth_map.py` | MiDaS depth estimation (torch.hub) | 68 |
| `generate_floor_mask.py` | SegFormer ADE20K floor segmentation + post-processing | 248 |
| `generate_configs.py` | Random JSON config generator | 153 |
| `add_pests_to_kitchen.py` | Core renderer: sprites + paths + COCO annotations → MP4 | 677 |
| `batch_render.py` | Parallel launcher for `add_pests_to_kitchen.py` | 147 |
| `extract_frames.py` | MP4 + sidecar JSON → JPEG frames + final COCO dataset | 337 |
| `benchmark.py` | Standalone timing harness | 475 |
| `config.json` | Example single-video config | 10 |
| `requirements.txt` | Pinned Python deps | 48 |

---

## 3. Full Pipeline Data Flow

```
kitchen.png
    │
    ▼ Step 1 ─── generate_depth_map.py ─────────────────────────────────
    │             MiDaS DPT_Large (torch.hub)
    │             Output: kitchen_depth.png  (grayscale, min-max normalised)
    │
    ▼ Step 2 ─── generate_floor_mask.py ────────────────────────────────
    │             nvidia/segformer-b2-finetuned-ade-512-512 (HuggingFace)
    │             Labels floor pixels (ADE20K label 3 = "floor")
    │             Post-process: fill holes → remove specks → Gaussian smooth
    │             Optional: AND with depth map (depth > thresh → floor)
    │             Output: kitchen_mask.png  (binary: 255 = walkable)
    │
    ▼ Step 3 ─── generate_configs.py ───────────────────────────────────
    │             Randomly samples pest counts, sizes, speeds, durations
    │             References kitchen.png + mask + depth paths
    │             Output: configs/config_0000.json … config_NNNN.json
    │
    ▼ Step 4 ─── batch_render.py → add_pests_to_kitchen.py (×N) ────────
    │             For each config JSON:
    │               • Loads background image + depth + floor mask
    │               • Instantiates PestAgent(s):
    │                   – generate_path(): random walk constrained to floor mask
    │                     Uses steering angle + speed smoothing + wall-bounce
    │                   – 32-attempt escape logic when stuck outside valid pixels
    │               • Per-frame loop:
    │                   – draw_mouse() / draw_cockroach() → BGRA sprite
    │                     (depth_at(x,y) → perspective scale 0.15–1.0)
    │                   – add_contact_shadow() (ellipse at feet)
    │                   – overlay_sprite() (alpha-composite onto background)
    │                   – sprite_to_bbox_and_mask() (alpha > 64 → binary mask)
    │                   – mask_to_polygon() or mask_to_rle() → segmentation
    │                   – Append to COCO images / annotations / frame_meta
    │               • Write MP4 via cv2.VideoWriter (mp4v codec)
    │               • Write <video>_coco.json alongside the MP4
    │             Output: videos/output_0000.mp4 + videos/output_0000_coco.json
    │
    ▼ Step 5 ─── extract_frames.py ─────────────────────────────────────
                  For each (MP4, _coco.json) pair:
                    • Index annotations by frame_idx
                    • OpenCV cap.read() frame by frame
                    • Skip every_n-1 frames if --every_n > 1
                    • Write JPEG (configurable quality, default 95)
                    • Collect (image_record, [ann_records]) tuples
                  shuffle + split all records → train / val / test
                  build_split_coco(): re-index image IDs + annotation IDs
                  Move JPEGs to images/{train,val,test}/
                  Write annotations/{train,val,test}.json
                  Write dataset_info.json (summary stats)

  Final output layout:
  out/<image_stem>/
  ├── kitchen_depth.png
  ├── kitchen_mask.png
  ├── configs/
  │   └── config_NNNN.json
  ├── videos/
  │   ├── output_0000.mp4
  │   └── output_0000_coco.json
  └── dataset/
      ├── images/
      │   ├── train/  *.jpg
      │   ├── val/    *.jpg
      │   └── test/   *.jpg
      ├── annotations/
      │   ├── train.json
      │   ├── val.json
      │   └── test.json
      └── dataset_info.json
```

---

## 4. Key Subsystems Explained

### 4.1 Depth Estimation (`generate_depth_map.py`)

- Loads MiDaS via `torch.hub.load("intel-isl/MiDaS", model)`.
- Applies the model-specific transform, normalises the 4-D batch dimension defensively.
- Runs inference, bicubic-upsamples back to original image size.
- **Min-max normalises to 0–255** and saves as grayscale PNG.

**What depth means here:** MiDaS outputs *relative inverse depth* — pixel brightness encodes distance from camera in a scene-relative, not metric, sense. Higher values tend to mean closer to the camera in most scenes.

### 4.2 Floor Mask (`generate_floor_mask.py`)

- SegFormer-B2 produces per-pixel ADE20K label predictions (150 classes).
- `labels_to_mask()` picks all pixels where label ∈ `floor_labels` (default: `[3]` = "floor").
- Three post-processing steps: `binary_fill_holes` (scipy) → connected-component speck removal → Gaussian boundary smoothing.
- **Safety checks prevent the mask from being destroyed:** if speck removal empties the mask, raw segmentation is used; if smoothing removes >20% of pixels, it is skipped.
- Optional depth refinement: `AND` the SegFormer mask with `depth_map > thresh` to discard depth-near (high brightness) pixels that SegFormer labeled as floor.

### 4.3 Sprite Drawing (`draw_mouse`, `draw_cockroach`)

Both functions draw entirely in code (no sprite sheets). They work on an `(c × c × 4)` BGRA canvas:
- Body/shell: overlapping circles along the spine axis
- Legs: sine-wave animated line segments per frame
- Tails/antennae: multi-segment curves with per-frame sway
- `frame_idx` drives all animations so motion is frame-consistent
- The sprite is then rotated by `angle_deg` (direction of travel, computed from 3-frame delta)

### 4.4 Perspective Scaling

```python
d = depth_at(self.depth_map, int(px), int(py))   # 0.0–1.0
persp_scale = float(np.clip(0.35 + 0.65 * d, 0.15, 1.0))
```

Pests at high-depth pixels (interpreted as farther from camera) are drawn larger, and those at low-depth pixels are drawn smaller. Since MiDaS depth is min-max normalised and relative, this gives a plausible perspective effect but is not geometrically accurate.

### 4.5 Path Generation (`generate_path`)

A first-order steering simulation:
- Random `target_angle` is re-sampled every 20–60 frames (steer timer)
- Maximum turn per frame: 8°
- Speed smoothly interpolates toward `target_speed` at 8% per frame
- Random pauses (0.8% chance per frame, 12–35 frames long)
- **Floor constraint:** if the next position is not in `valid_set`, up to 32 escape attempts are made with increasing angular spread; if all fail, teleport to nearest valid pixel

### 4.6 COCO Annotations

Each frame generates:
- `images[]`: `{id, file_name, width, height, frame_idx, timestamp, video}`
- `annotations[]`: `{id, image_id, category_id, bbox, area, segmentation, iscrowd, track_id}`
  - `bbox` is tight around the alpha-channel mask (not a loose box around the sprite canvas)
  - `segmentation` is polygon if contours are found, otherwise falls back to uncompressed RLE
  - `track_id` is stable across all frames for each agent (1-indexed)
- `frame_meta[]`: per-frame `{has_pest, pest_count}` for classifier label extraction

### 4.7 Dataset Assembly (`extract_frames.py`)

- Frame records from all videos are pooled, shuffled (with seed), and split by fraction.
- IDs are globally re-indexed (1-based) to be unique across the merged dataset.
- JPEGs are moved from a temp directory to the final split folder; `file_name` is rewritten to relative paths (`images/train/...`).

---

## 5. Configuration Schema

```json
{
  "image":        "kitchen1.png",       // required: background image
  "mask":         "kitchen1_mask.png",  // optional: floor mask PNG
  "depth":        "kitchen1_depth.png", // optional: depth map PNG
  "output":       "output.mp4",         // output video path
  "duration":     30,                   // seconds
  "fps":          25,
  "floor_thresh": 30,                   // 0-255; depth fallback threshold
  "pests": [
    { "type": "mouse",     "count": 1, "size": 50, "speed": 6 },
    { "type": "cockroach", "count": 2, "size": 30, "speed": 9 }
  ]
}
```

---

## 6. Critical Issues

### 6.1 🔴 Non-Standard RLE Encoding (Will Break Most Training Frameworks)

**File:** `add_pests_to_kitchen.py`, `mask_to_rle()`

The function produces an **uncompressed integer-array RLE**:
```python
{"counts": [0, 5, 10, 3, ...], "size": [H, W]}
```

The COCO spec (and all standard tools — pycocotools, Detectron2, MMDetection, YOLO) expect **byte-encoded compressed RLE** produced by `pycocotools.mask.encode()`. If polygon conversion succeeds (most cases), this is fine because polygons are used preferentially. But when `mask_to_polygon()` returns `None` (e.g. for degenerate sprites near the image edge), the uncompressed RLE is stored and **will cause silent parse failures or crashes** when loading with pycocotools.

**Fix:** Use `pycocotools.mask.encode(np.asfortranarray(binary_mask))` and decode `counts` as a string (base-64) for the sidecar JSON.

---

### 6.2 🔴 `generate_depth_map.py` Has Module-Level Argparse

**File:** `generate_depth_map.py`, lines 9–17

```python
parser = argparse.ArgumentParser(...)
parser.add_argument(...)
args = parser.parse_args()   # ← runs at import time
```

There is no `main()` function and no `if __name__ == "__main__":` guard. This means:
- The file **cannot be imported** as a module from any other script without crashing.
- `run_pipeline.py` works around this by always calling it via `subprocess.run`, but any attempt to `import generate_depth_map` will immediately call `sys.argv` parsing and fail.

All other scripts in the project follow the correct `def main() / if __name__ == "__main__":` pattern.

**Fix:** Wrap everything inside `def main()` and add the guard.

---

### 6.3 🔴 `.jpg` / `.jpeg` Input Breaks Depth Map Output Path

**File:** `generate_depth_map.py`, line 19

```python
output_path = args.output if args.output else args.image.replace(".png", "_depth.png")
```

If `args.image` is `kitchen.jpg`, `.replace(".png", "_depth.png")` does nothing — `output_path` becomes `kitchen.jpg`, **overwriting the input image** with a grayscale depth map.

**Fix:**
```python
from pathlib import Path
output_path = args.output or str(Path(args.image).with_suffix("")) + "_depth.png"
```

---

### 6.4 🟠 MiDaS Depth Semantics vs. Floor Mask Refinement Are Inverted

**Files:** `generate_floor_mask.py` `refine_with_depth()`, `add_pests_to_kitchen.py` lines 550–551

MiDaS outputs **relative inverse depth**: brighter pixels (value → 1.0) are *closer* to the camera, not farther. A kitchen floor at the bottom of the frame is typically farther from the camera than countertops and cabinets — so floor pixels often have *lower* depth values.

The code uses:
```python
# generate_floor_mask.py
refined = np.where(depth > thresh_norm, mask, 0)   # keeps HIGH depth = close objects

# add_pests_to_kitchen.py fallback (no mask provided)
floor_mask = depth_map > floor_thresh              # also selects HIGH depth
```

In many kitchen images this would **select counters and cabinets as the walkable floor**, not the actual floor. The pest perspective scaling also becomes counter-intuitive: pests near-camera (high depth, low actual distance) are drawn larger, which happens to be geometrically correct by accident for some scenes but wrong for others.

**Fix:** Invert the depth comparison: `depth < thresh_norm` selects floor (far from camera = low MiDaS value), or flip the depth map: `depth_map = 1.0 - depth_map` before any thresholding. Also reconsider the `persp_scale` formula.

---

### 6.5 🟠 Batch Render Swallows All Render Output

**File:** `batch_render.py`, line 62

```python
result = subprocess.run(
    [sys.executable, script, "--config", tmp_cfg],
    capture_output=True, text=True, timeout=600   # ← captures stdout+stderr
)
```

With `capture_output=True`, all frame-by-frame progress from `add_pests_to_kitchen.py` is invisible. On failure, only the last line of stderr is reported:
```python
err = result.stderr.strip().split("\n")[-1]
```

This makes debugging render failures very hard — you see one line of context for a 10-minute job.

**Fix:** Remove `capture_output=True` for real-time display, or save the full output to a log file per config, or at minimum print the full `result.stderr` on failure.

---

### 6.6 🟠 Dead Stub Function in `generate_configs.py`

**File:** `generate_configs.py`, lines 58–61

```python
def generate_config(image, mask, depth, output_video, duration, fps):
    """Build one config dict with a random mix of pests."""
    # Randomly decide how many of each pest (caller already sampled counts)
    return None  # built by caller
```

The function exists, is named as if it does something, has a docstring, but unconditionally returns `None`. All actual config building is done inline in `main()`. This is confusing dead code.

**Fix:** Either implement the function properly and call it from `main()`, or delete it.

---

### 6.7 🟡 No GPU Utilisation in Either Deep Learning Step

**Files:** `generate_depth_map.py`, `generate_floor_mask.py`

Neither script moves the model or input tensors to a GPU:
```python
# generate_depth_map.py — model always runs on CPU
midas = torch.hub.load("intel-isl/MiDaS", args.model)

# generate_floor_mask.py — same
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()
```

On CPU, `DPT_Large` can take 30–120 seconds per image and SegFormer-B2 takes 10–30 seconds. With a GPU these drop to under 2 seconds.

**Fix:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_batch = input_batch.to(device)
```

---

### 6.8 🟡 `build_split_coco` Computes `old_id` But Never Uses It

**File:** `extract_frames.py`, lines 143–151

```python
for img_record, anns in all_records:
    old_id = img_record["image_id"] if "image_id" in img_record else img_record["id"]
    new_img = {k: v for k, v in img_record.items()}
    new_img["id"] = new_image_id
    coco["images"].append(new_img)

    for ann in anns:
        new_ann = {k: v for k, v in ann.items()}
        new_ann["id"]       = new_ann_id
        new_ann["image_id"] = new_image_id  # ← ignores old_id, uses new_image_id directly
```

`old_id` is computed and then completely ignored. The annotation-to-image remapping works correctly only because `all_records` was pre-assembled as paired `(img_record, [anns])` tuples, so no lookup by ID is needed. However, the unused variable is misleading and implies a remapping step that does not exist.

Additionally, `add_pests_to_kitchen.py` stores `"id": i` on images (not `"image_id"`), so the check `"image_id" in img_record` is always `False` — yet another sign of uncoordinated field naming.

**Fix:** Delete the `old_id` line; the code is correct without it.

---

### 6.9 🟡 `img_record` Mutation During Dataset Assembly

**File:** `extract_frames.py`, lines 293–296

```python
for img_rec, _ in records:
    src = tmp_dir / img_rec["file_name"]
    dst = dest_img_dir / img_rec["file_name"]
    if src.exists():
        shutil.move(str(src), str(dst))
    img_rec["file_name"] = f"images/{split_name}/{img_rec['file_name']}"   # ← mutates dict
```

`split_records()` returns a shallow copy of the list, so each `img_rec` dict is the **same object** held in `all_records`. If anything iterated `all_records` again after this loop (currently nothing does), all `file_name` values would already be prefixed. This is fragile.

**Fix:** Use `new_img = {**img_rec}` before mutating, or build a fresh dict in `build_split_coco`.

---

### 6.10 🟡 `floor_thresh` and `depth_thresh` Are Named Inconsistently

The same concept (depth cutoff for floor detection) is called `floor_thresh` (0–255 in config, divided by 255 in `add_pests_to_kitchen.py`) and `depth_thresh` (0–255, passed directly as a float) in `run_pipeline.py` and `generate_floor_mask.py`. They operate on the same scale but are named differently in every script, creating confusion when reading configs or comparing pipeline flags.

---

### 6.11 🟡 Perspective Scale Logic Is Inverted for Most Scenes

**File:** `add_pests_to_kitchen.py`, lines 485–489

```python
if self.depth_map is not None:
    d = depth_at(self.depth_map, int(px), int(py))
else:
    d = py / img_h   # fallback: lower pixel row = larger pest

persp_scale = float(np.clip(0.35 + 0.65 * d, 0.15, 1.0))
```

When a depth map is available, higher `d` → larger pest. Since MiDaS high values = close to camera, pests near the camera (foreground) are drawn larger. In a typical kitchen perspective this is *geometrically wrong*: the floor foreground (close camera) has *low* MiDaS depth values. The fallback (`py / img_h`) is also the opposite of the depth-map convention for a kitchen scene.

This is a direct consequence of Issue 6.4 — fixing the depth inversion there will also fix this.

---

## 7. Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 6.1 | 🔴 Critical | `add_pests_to_kitchen.py` | Non-standard RLE will break pycocotools |
| 6.2 | 🔴 Critical | `generate_depth_map.py` | Module-level argparse — file un-importable |
| 6.3 | 🔴 Critical | `generate_depth_map.py` | `.jpg` input overwrites source image |
| 6.4 | 🟠 High | `generate_floor_mask.py`, `add_pests_to_kitchen.py` | MiDaS depth semantics inverted vs. thresholding logic |
| 6.5 | 🟠 High | `batch_render.py` | `capture_output=True` hides all render progress/errors |
| 6.6 | 🟠 Medium | `generate_configs.py` | `generate_config()` is a dead stub always returning `None` |
| 6.7 | 🟡 Medium | `generate_depth_map.py`, `generate_floor_mask.py` | No GPU — depth + mask generation very slow |
| 6.8 | 🟡 Low | `extract_frames.py` | `old_id` computed but never used |
| 6.9 | 🟡 Low | `extract_frames.py` | `img_rec["file_name"]` mutated in-place |
| 6.10 | 🟡 Low | All scripts | `floor_thresh` vs `depth_thresh` naming inconsistency |
| 6.11 | 🟡 Medium | `add_pests_to_kitchen.py` | Perspective scale direction is inverted for typical scenes |

---

## 8. What Works Well

- **Annotation fidelity:** Bounding boxes and polygon masks are derived directly from the alpha channel of the rendered sprite — they are **pixel-perfect** and zero-error by construction.
- **Track IDs:** Stable `track_id` across all frames enables tracking-model training, not just detection.
- **Floor-constrained paths:** The 32-attempt escape + nearest-valid-pixel teleport prevents pests from getting permanently stuck without breaking the video.
- **Frame metadata:** `frame_meta[]` with `has_pest` / `pest_count` allows the same dataset to be used for binary classification (pest / no-pest) in addition to detection.
- **Pipeline skip flags:** `--skip_depth`, `--skip_mask`, `--skip_configs`, `--skip_extract` make iterative re-runs fast.
- **Safety rails in floor mask post-processing:** The "never remove >20% of floor" and "if empty use raw" guards prevent catastrophic mask failures.
- **Deterministic reproducibility:** Every random element (paths, configs, dataset split) accepts a `--seed` argument.
