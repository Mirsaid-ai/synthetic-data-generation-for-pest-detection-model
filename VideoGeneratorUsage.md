# Synthetic Pest Video Generator

Generates synthetic kitchen videos with animated pests (mice, cockroaches) for training computer vision models. Outputs MP4 videos alongside COCO JSON annotation files with bounding boxes, instance segmentation masks, and tracking IDs. A frame extraction step assembles everything into a train/val/test dataset.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `generate_depth_map.py` | Depth estimation using MiDaS DPT_Large |
| `generate_floor_mask.py` | Segments floor from kitchen image via SegFormer |
| `generate_configs.py` | Randomly generates video config JSON files |
| `add_pests_to_kitchen.py` | Renders one video + `_coco.json` from a config |
| `batch_render.py` | Renders all configs in a directory |
| `extract_frames.py` | Extracts frames + assembles COCO train/val/test dataset |
| `run_pipeline.py` | All-in-one: depth → mask → configs → videos → dataset |
| `benchmark.py` | Measures render speed and projects scale costs |

---

## Installation

```bash
pip install torch torchvision transformers pillow opencv-python numpy scipy timm
```

> `torch`, `transformers`, and `timm` are only needed for `generate_depth_map.py` and `generate_floor_mask.py`. Everything else only requires `opencv-python` and `numpy`.

---

## Quick Start

### Option A — Single config (recommended starting point)

```bash
python run_pipeline.py --config config.json --output_dir out/
```

`--image` is optional when using `--config` — the pipeline reads the image path directly from the config file. Runs: depth map → floor mask → render video + COCO annotations → extract frames → dataset.

### Option B — Random video generation

```bash
python run_pipeline.py \
  --image kitchen1.png \
  --n 20 \
  --mice 0 3 \
  --cockroaches 0 5 \
  --duration 15 30 \
  --output_dir out/
```

Generates 20 random configs, renders all of them, and builds a dataset.

### Option C — Multiple images without collisions

Each image gets its own scoped subdirectory under `--output_dir`, so running on different images never overwrites anything:

```bash
python run_pipeline.py --config config_kitchen1.json --output_dir out/
python run_pipeline.py --config config_kitchen3.json --output_dir out/
```

Produces `out/kitchen1/` and `out/kitchen3/` fully independently.

### Option D — Step by step

```bash
# 1. Generate depth map
python generate_depth_map.py --image kitchen1.png --output kitchen1_depth.png

# 2. Generate floor mask (first run downloads ~110MB model)
python generate_floor_mask.py \
  --image kitchen1.png \
  --depth kitchen1_depth.png \
  --output kitchen1_mask.png \
  --debug

# 3. Generate random configs
python generate_configs.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --depth kitchen1_depth.png \
  --output_dir configs/ \
  --n 20 \
  --mice 0 3 \
  --cockroaches 0 5

# 4. Render all configs (each produces a .mp4 + _coco.json sidecar)
python batch_render.py \
  --config_dir configs/ \
  --output_dir videos/ \
  --jobs 4

# 5. Extract frames and build dataset
python extract_frames.py \
  --video_dir videos/ \
  --output_dir dataset/ \
  --split 0.8 0.1 0.1 \
  --every_n 3

# Or render a single config directly
python add_pests_to_kitchen.py --config config.json
```

---

## Output Structure

Each image gets its own directory under `--output_dir`, so multiple images never collide:

```
out/
├── kitchen1/                        # scoped to kitchen1.png
│   ├── kitchen1_depth.png
│   ├── kitchen1_mask.png
│   ├── kitchen1_mask_debug.png      # with --debug_mask
│   ├── configs/
│   │   ├── config_0000.json
│   │   └── ...
│   ├── videos/
│   │   ├── kitchen1_output.mp4
│   │   ├── kitchen1_output_coco.json  # per-video COCO annotations
│   │   └── ...
│   └── dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── annotations/
│       │   ├── train.json             # merged COCO JSON
│       │   ├── val.json
│       │   └── test.json
│       └── dataset_info.json
└── kitchen3/                        # scoped to kitchen3.png
    └── ...
```

---

## Config File Format

When running via `run_pipeline.py`, do **not** include `mask` or `depth` paths in your config — the pipeline computes and injects these automatically. Only include them if running `add_pests_to_kitchen.py` directly.

```json
{
  "image":    "kitchen1.png",
  "output":   "output.mp4",
  "duration": 30,
  "fps":      25,
  "pests": [
    { "type": "mouse",     "count": 1, "size": 50, "speed": 6 },
    { "type": "cockroach", "count": 3, "size": 30, "speed": 9 }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `image` | ✅ | Kitchen background image |
| `mask` | Auto-injected by pipeline | Floor mask PNG — set manually only when running standalone |
| `depth` | Auto-injected by pipeline | Depth map — set manually only when running standalone |
| `output` | Optional | Output video filename (default: `output.mp4`) |
| `duration` | Optional | Video length in seconds (default: 10) |
| `fps` | Optional | Frames per second (default: 25) |
| `pests` | ✅ | List of pest entries |

### Pest entry fields

| Field | Description | Default |
|---|---|---|
| `type` | `"mouse"` or `"cockroach"` | required |
| `count` | Number of this pest | 1 |
| `size` | Base sprite size in pixels | 50 |
| `speed` | Movement speed in pixels/frame | 6 |

---

## COCO Annotations

Each rendered video automatically produces a `_coco.json` sidecar. `extract_frames.py` then merges these into standard train/val/test COCO JSON files.

### Annotation fields

| Field | Description |
|---|---|
| `id` | Unique annotation ID |
| `image_id` | Links to the image record |
| `category_id` | `1` = mouse, `2` = cockroach |
| `bbox` | `[x, y, w, h]` tight bounding box (pixel-accurate from sprite alpha) |
| `area` | Pixel area of the instance mask |
| `segmentation` | Polygon contours or RLE — instance segmentation mask |
| `iscrowd` | Always `0` |
| `track_id` | Stable integer per pest across all frames of a video |

### Classifier labels

The `frame_meta` array in the per-video `_coco.json` provides frame-level classifier labels:

```json
{
  "frame_idx": 42,
  "has_pest": true,
  "pest_count": 3,
  "file_name": "kitchen1_output_frame_000042.jpg"
}
```

### Categories

```json
[
  {"id": 1, "name": "mouse",     "supercategory": "pest"},
  {"id": 2, "name": "cockroach", "supercategory": "pest"}
]
```

### Example annotation

```json
{
  "id": 17,
  "image_id": 5,
  "category_id": 2,
  "bbox": [312.0, 418.0, 28.0, 22.0],
  "area": 412.0,
  "segmentation": [[318, 418, 320, 419, ...]],
  "iscrowd": 0,
  "track_id": 2
}
```

---

## Script Reference

### `generate_depth_map.py`

```bash
python generate_depth_map.py --image kitchen1.png --output kitchen1_depth.png
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input kitchen image |
| `--output` | `<image>_depth.png` | Output depth map path |
| `--model` | `DPT_Large` | MiDaS model: `DPT_Large`, `DPT_Hybrid`, `MiDaS_small` |

`DPT_Large` gives the best quality. Use `MiDaS_small` if you need faster generation at lower quality.

---

### `generate_floor_mask.py`

```bash
python generate_floor_mask.py --image kitchen1.png --output kitchen1_mask.png --debug
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input kitchen image |
| `--output` | `floor_mask.png` | Output binary mask PNG |
| `--depth` | None | Depth map to AND with segmentation result |
| `--depth_thresh` | 40 | Depth cutoff 0–255 (pixels above = floor) |
| `--floor_labels` | `3` | ADE20K label indices for floor |
| `--smooth_px` | 5 | Boundary smoothing radius (0 to disable) |
| `--debug` | off | Save a colour-coded overlay PNG |

**ADE20K floor labels:** `3`=floor, `6`=road, `52`=rug

| Symptom | Fix |
|---|---|
| Mask is completely black | Remove `--depth` or set `--depth_thresh 10` |
| Mask misses floor near walls | Lower `--smooth_px 2` or set `--smooth_px 0` |
| Wrong region detected | Check debug overlay; add `--floor_labels 3 6` |

---

### `generate_configs.py`

```bash
python generate_configs.py \
  --image kitchen1.png \
  --mask kitchen1_mask.png \
  --output_dir configs/ \
  --n 50 --mice 0 3 --cockroaches 0 5 --duration 15 30 --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Kitchen image path |
| `--mask` | None | Floor mask path |
| `--depth` | None | Depth map path |
| `--output_dir` | `configs/` | Directory to write JSON configs |
| `--n` | 10 | Number of configs to generate |
| `--mice` | `0 3` | Min and max mice per video |
| `--cockroaches` | `0 5` | Min and max cockroaches per video |
| `--duration` | `15 30` | Duration range in seconds |
| `--fps` | 25 | Frames per second |
| `--seed` | None | Random seed for reproducibility |

---

### `batch_render.py`

```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs 4
```

| Flag | Default | Description |
|---|---|---|
| `--config_dir` | required | Directory with JSON config files |
| `--output_dir` | alongside configs | Where to save MP4 videos |
| `--jobs` | 1 | Parallel render processes |
| `--no_skip` | off | Re-render even if video already exists |
| `--fail_fast` | off | Stop on first render error |

---

### `extract_frames.py`

```bash
python extract_frames.py \
  --video_dir videos/ \
  --output_dir dataset/ \
  --split 0.8 0.1 0.1 \
  --every_n 3 \
  --no_empty
```

| Flag | Default | Description |
|---|---|---|
| `--video_dir` | — | Directory with `.mp4` + `_coco.json` pairs |
| `--video` | — | Single video (alternative to `--video_dir`) |
| `--output_dir` | `dataset/` | Root dataset output directory |
| `--split` | `0.8 0.1 0.1` | Train/val/test fractions (must sum to 1) |
| `--quality` | 95 | JPEG quality 1–100 |
| `--every_n` | 1 | Extract every Nth frame (reduces redundancy) |
| `--no_empty` | off | Skip frames with no pest annotations |
| `--seed` | 42 | Random seed for split |

---

### `run_pipeline.py`

```bash
# Single config — no --image needed
python run_pipeline.py --config config.json --output_dir out/

# Random generation — --image required
python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/
```

| Flag | Default | Description |
|---|---|---|
| `--image` | from config | Kitchen image. Read from config if `--config` is provided |
| `--output_dir` | `pipeline_out/` | Root output directory. Each image gets its own subdirectory |
| `--config` | None | Single config JSON — skips random generation |
| `--n` | 10 | Number of random videos (random generation mode only) |
| `--mice` | `0 3` | Mice count range |
| `--cockroaches` | `0 5` | Cockroach count range |
| `--duration` | `15 30` | Video duration range (seconds) |
| `--fps` | 25 | Frames per second |
| `--jobs` | 1 | Parallel render jobs |
| `--floor_labels` | `3` | ADE20K floor labels for mask generation |
| `--depth_thresh` | 40 | Depth threshold for mask generation |
| `--split` | `0.8 0.1 0.1` | Dataset train/val/test split |
| `--every_n` | 1 | Extract every Nth frame for dataset |
| `--no_empty_frames` | off | Skip frames with no pests in dataset |
| `--skip_depth` | off | Skip depth generation if file already exists |
| `--skip_mask` | off | Skip mask generation if file already exists |
| `--skip_configs` | off | Skip config generation if configs dir has files |
| `--skip_extract` | off | Skip frame extraction and dataset assembly |
| `--debug_mask` | off | Save debug overlay for the floor mask |
| `--seed` | None | Random seed for configs and dataset split |

---

### `benchmark.py`

```bash
# Quick estimate
python benchmark.py --quick

# Full benchmark with your actual files
python benchmark.py \
  --image kitchen1.png \
  --mask out/kitchen1/kitchen1_mask.png \
  --depth out/kitchen1/kitchen1_depth.png \
  --duration 30 --fps 25 --runs 3
```

| Flag | Default | Description |
|---|---|---|
| `--image` | synthetic 1920×1080 | Kitchen image |
| `--mask` | synthetic | Floor mask |
| `--depth` | synthetic gradient | Depth map |
| `--duration` | 10 | Video duration per benchmark run (seconds) |
| `--fps` | 25 | Frames per second |
| `--runs` | 3 | Timing runs to average per scenario |
| `--quick` | off | 5-second single run for fast estimate |
| `--output` | None | Save a sample rendered video |

Benchmarks four scenarios (1 mouse, 3 cockroaches, 2 mice + 3 cockroaches, 5 cockroaches). Reports ms/frame, achieved fps, and projects wall-clock time for 100 / 1,000 / 10,000 videos with parallelism estimates based on your core count.

---

## Tips

**Running on multiple images**
```bash
# Each image is fully isolated — nothing is overwritten
python run_pipeline.py --config config_kitchen1.json --output_dir out/
python run_pipeline.py --config config_kitchen3.json --output_dir out/
```

**Skipping steps you've already run**
```bash
python run_pipeline.py --config config.json --output_dir out/ \
  --skip_depth --skip_mask
```

**Speeding up large batches**
```bash
python batch_render.py --config_dir configs/ --output_dir videos/ --jobs $(nproc)
```

**Resuming interrupted batches**
`batch_render.py` skips already-rendered videos by default. Just re-run the same command.

**Reducing dataset size without losing diversity**
Use `--every_n 3` or `--every_n 5`. This removes temporal redundancy (adjacent frames look nearly identical) while preserving the full range of motion.

**Training with Ultralytics YOLO**
```bash
yolo detect train data=out/kitchen1/dataset/annotations/train.json model=yolov8n.pt
```

**Training with Detectron2**
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("pests_train", {},
    "out/kitchen1/dataset/annotations/train.json",
    "out/kitchen1/dataset/images/train")
register_coco_instances("pests_val", {},
    "out/kitchen1/dataset/annotations/val.json",
    "out/kitchen1/dataset/images/val")
```

**Tuning pest appearance**
In `add_pests_to_kitchen.py`, sprite proportions are multiples of `r` (body radius) inside `draw_mouse()` and `draw_cockroach()`:

| Parameter | Mouse | Cockroach |
|---|---|---|
| Body radius ratio | `c * 0.22` | `c * 0.20` |
| Body length | `r * 1.6` | `r * 1.8` |
| Head size | `r * 0.68` | `r * 0.45` |
| Antenna length | — | `r * 3.0` |

---

## How Annotations Are Generated

Annotations are computed during the render pass with zero overhead — no post-processing step required.

For each frame and each pest:
1. The sprite is drawn to an off-screen BGRA canvas
2. The **alpha channel** of the sprite is used as the pixel-perfect instance mask
3. A tight **bounding box** is extracted from the mask extents
4. **Polygon contours** are traced from the alpha mask for segmentation
5. The **track ID** is a stable integer assigned at pest creation time, unchanged across all frames

Annotations are ground-truth accurate — no labelling error, no occlusion ambiguity, no annotation noise.

## TODO

1. add feature that allows pests to disappear below counter tops
2. add pest spawning on flat surfaces other than the floor
3. change in brightness as pest enters or exits shadow
