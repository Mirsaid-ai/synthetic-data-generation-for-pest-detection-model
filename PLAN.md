# Training Plan — Synthetic Pest Detection
> Goal: ≥80% true detection rate, <5% false positive rate on real video
> Target environment: NC State University cafeteria kitchen (security/CCTV camera, ceiling-mounted)
> Last updated: April 2026

---

## Performance Target (Locked)

| Metric | Requirement |
|--------|-------------|
| True detection rate (recall) | ≥ 80% |
| False positive rate (FPR) | < 5% |
| Evaluation unit | Per-frame on test video |
| Test data | Real kitchen video (NC State cafeteria, camera TBD) |

> **Note:** Both constraints must hold simultaneously at a single confidence threshold.
> FPR < 5% is the harder constraint — it requires explicit negative training data.
> **Critical:** Train/val/test split must be at the kitchen level (not frame level) to avoid data leakage.

---

## Compute & Storage Resources

| Resource | Availability | Role |
|----------|-------------|------|
| Local MacBook M3 Pro (MPS) | Available | Prototyping, depth pre-computation, light runs |
| Google Colab Pro (T4, 16GB) | Available | Primary training + batch rendering |
| Google Drive 2TB | Available | Dataset storage (pipeline ↔ Colab) |
| Duke Cluster | ❌ No access | — |
| Real video (NC State) | Not yet — to be obtained | Fine-tuning (high priority when available) |

---

## Current Asset Inventory

| Asset | Count | Status |
|-------|-------|--------|
| Curated kitchen images (upscaled) | 49 | ✅ Upscaled 256×256 → 1024×1024 via Real-ESRGAN |
| Gemini-generated kitchen images | ~100 | 🔄 Generating (overhead/CCTV angle, counter surfaces) |
| Sagnik sprites (mouse/rat/cockroach) | ~13/type | ✅ Ready |
| Generator pipeline | `generator/` | ✅ Merged + bug-fixed + CCTV simulation |
| CCTV simulation | `generator/compositing.py` | ✅ Done |
| Pre-computed depth cache | `depth_cache/` | ❌ Run `precompute_depths.py` |
| DETR training code | `model/` | ⚠️ Replaced by YOLOv8 |

---

## Step-by-Step Plan

---

### Step 1 — Upscale Existing Kitchen Images ✅ DONE
**Where:** Local M3 Pro
**Script:** `scripts/upscale_images.py`

Used Real-ESRGAN 4× to upscale 49 curated images from 256×256 → 1024×1024.

---

### Step 2 — Generate New Commercial Kitchen Images 🔄 IN PROGRESS
**Where:** Local (Gemini API)
**Script:** `scripts/generate_kitchen_images.py`

Generating ~100 images with overhead/CCTV-angle prompts.
Staging area: `generator/kitchen_img/generated_img/` → review → move to `curated_img/`

**Prompt priorities (by weight):**
- Weight 3: Overhead/CCTV angle (most important — matches test environment)
- Weight 2: Counter/table surfaces (pests spawn on counters too)
- Weight 1: Diverse angles/lighting for domain diversity + IR night-mode variant

**After review:** Move keepers to `generator/kitchen_img/curated_img/`

**Target total after Steps 1+2:** ~130–150 kitchen images at ≥1024×1024

---

### Step 3 — CCTV Simulation in Video Compositor ✅ DONE
**Where:** `generator/compositing.py`

Added `CCTVSimulator` class — sampled **once per video** (consistent within a clip):

| Effect | Range | Coverage |
|--------|-------|----------|
| Gaussian sensor noise | σ = 5–20 | 100% of videos |
| JPEG compression artefacts | quality = 60–80 | 100% of videos |
| Brightness variation | factor = 0.7–1.3 | 100% of videos |
| IR / grayscale night mode | — | 30% of videos |
| Resolution halve + upsample | ÷2 then ×2 | 20% of videos |
| Motion blur on sprites | kernel ∝ pest velocity | 100% of videos (per-frame) |

---

### Step 4 — Fix Critical Bugs ✅ DONE
**Completed:** April 2026

| # | File | Status |
|----|------|--------|
| 1 | `add_pests_to_kitchen.py` line 558 | ✅ Fixed — depth inversion |
| 2 | `add_pests_to_kitchen.py` `mask_to_rle()` | ✅ Already correct |
| 3 | `generate_depth_map.py` JPG input | ✅ Already correct |
| 4 | `generate_floor_mask.py` depth semantics | ✅ Already correct |
| 5 | `generator/pipeline.py` depth leakage | ✅ Fixed — precomputed_depth param added |
| 6 | `extract_frames.py` frame-level split | ✅ Fixed — replaced by `build_dataset.py` with kitchen-level split |

---

### Step 4b — Pre-compute Depth Maps ← DO THIS BEFORE COLAB
**Where:** Local M3 Pro (MPS-accelerated)
**Script:** `scripts/precompute_depths.py`
**Time:** ~30–60 minutes for 150 images

Metric3D v2 on Colab CPU takes 5–20 min per image. With 20 videos per kitchen,
this would waste 95%+ of Colab compute. Pre-computing locally reduces the Colab
depth step from minutes → milliseconds (just a numpy load).

```bash
python scripts/precompute_depths.py
# → saves depth_cache/*.npz (one per kitchen image)
# → upload depth_cache/ to Google Drive
```

---

### Step 5 — Batch Render 3,000+ Videos
**Where:** Google Colab CPU (5–6 parallel sessions) → Google Drive
**Script:** `scripts/render_batch_colab.py`
**Time:** ~4–6 hours wall time

**Kitchen-level split (enforced before rendering):**

| Split | Kitchens | Videos | Purpose |
|-------|----------|--------|---------|
| Train | ~120 (80%) | ~2,400 | Training data |
| Val | ~15 (10%) | ~300 | Validation during training |
| Held-out | ~15 (10%) | 0 (not rendered) | Final test evaluation |

**Video configuration:**

| Parameter | Value |
|-----------|-------|
| Resolution | 640×480 |
| FPS | 10 |
| Duration | 30 seconds (300 frames, save every 2nd = 150/video) |
| Videos per kitchen | 20 |
| Pest mix | 25% zero-pest, 30% single, 45% multi-pest |

**Colab setup:**
```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q torch torchvision pillow opencv-python scipy

!python scripts/render_batch_colab.py \
    --image_dir   /content/drive/MyDrive/pest_project/kitchens \
    --depth_cache /content/drive/MyDrive/pest_project/depth_cache \
    --output_dir  /content/drive/MyDrive/pest_project/renders \
    --n 20 \
    --session_id 0 --total_sessions 5
```
Run 5 sessions simultaneously with `--session_id 0` through `--session_id 4`.

---

### Step 6 — Build YOLO Dataset (Kitchen-Level Split)
**Where:** Local or Colab
**Script:** `scripts/build_dataset.py`
**Time:** ~30 minutes

Reads the render manifest (which has the kitchen split baked in), converts
COCO JSON annotations → YOLOv8 `.txt` label files, writes `data.yaml`.

```bash
python scripts/build_dataset.py \
    --render_dir /drive/renders \
    --output_dir /drive/pest_dataset \
    --every_n 2        # use every 2nd frame → halves storage, still ~170K frames
```

**Output structure:**
```
pest_dataset/
├── images/
│   ├── train/   (~136,000 frames)
│   ├── val/     (~17,000 frames)
│   └── test/    (from held-out render — Step 6b)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**Step 6b — Generate test set from held-out kitchens:**
```bash
python scripts/render_batch_colab.py \
    --image_dir /drive/kitchens --depth_cache /drive/depth_cache \
    --output_dir /drive/renders_held_out --n 20 --held_out_mode

python scripts/build_dataset.py \
    --render_dir /drive/renders \
    --test_render_dir /drive/renders_held_out \
    --output_dir /drive/pest_dataset
```

---

### Step 7 — Train YOLOv8m on Google Colab Pro
**Where:** Google Colab Pro (T4, 16GB)
**Time:** ~6–8 hours

```bash
yolo train \
  data=/content/drive/MyDrive/pest_dataset/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=/content/drive/MyDrive/pest_runs \
  name=yolov8m_pest_v1 \
  augment=True \
  degrees=10 \
  fliplr=0.5 \
  hsv_h=0.015 \
  hsv_s=0.5 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.3 \
  mosaic=1.0
```

Checkpoints saved to Google Drive — safe against Colab disconnects.

---

### Step 8 — Threshold Calibration + Evaluation
**Where:** Local M3 Pro
**Script:** `scripts/calibrate_threshold.py` (to be written)
**Time:** ~1–2 hours

**Test sets:**
1. **Synthetic test** — held-out kitchens, never seen in training
2. **Negative test** — 200+ frames of empty kitchen from held-out kitchens

**Process:**
1. Run inference on all test frames, collect raw confidence scores
2. Sweep confidence threshold 0.05 → 0.95 in steps of 0.05
3. At each threshold: compute recall and FPR
4. Find threshold T where: recall(T) ≥ 0.80 AND FPR(T) < 0.05
5. If no such T exists → return to Step 7 (more epochs or more negative data)

---

### Step 9 — Real Video Fine-Tuning (When Video Is Available)
**Where:** Google Colab Pro
**Time:** ~1–2 hours labeling + ~1 hour training

Even 200–500 labeled real frames can close the sim-to-real gap.

**Labeling tool:** [Label Studio](https://labelstud.io) — free, local, exports YOLO format.

```bash
yolo train model=/path/to/best.pt data=real_data.yaml \
  epochs=30 lr0=0.0001 batch=8
```

---

## Complete Workflow Sequence

```
Local M3 Pro:
  ① scripts/generate_kitchen_images.py --count 100 --staging    (running)
  ② Review generated_img/, move keepers to curated_img/
  ③ scripts/precompute_depths.py                                 (~1 hr)
  ④ Upload to Google Drive:
       depth_cache/  generator/  kitchen images  sprites/

Colab (5 parallel sessions):
  ⑤ scripts/render_batch_colab.py  --session_id 0..4            (~4-6 hrs)

Local or Colab:
  ⑥ scripts/build_dataset.py        → pest_dataset/
  ⑦ yolo train  data=data.yaml ...  → best.pt                   (~6-8 hrs)
  ⑧ scripts/calibrate_threshold.py  → find optimal T

Optional (when real video available):
  ⑨ Label 300-500 real frames with Label Studio
     yolo train  model=best.pt  epochs=30  lr0=0.0001
```

---

## Scripts Reference

| Script | Purpose | Step |
|--------|---------|------|
| `scripts/upscale_images.py` | Real-ESRGAN 4× upscale of kitchen images | 1 ✅ |
| `scripts/generate_kitchen_images.py` | Gemini API batch image generation | 2 ✅ |
| `scripts/precompute_depths.py` | Metric3D depth cache — run locally before Colab | 4b ✅ |
| `scripts/render_batch_colab.py` | Kitchen-level split + batch rendering for Colab | 5 ✅ |
| `scripts/build_dataset.py` | COCO→YOLO conversion + kitchen-level dataset split | 6 ✅ |
| `scripts/calibrate_threshold.py` | Sweep threshold, find optimal T for recall/FPR | 8 ❌ |

---

## Key Decisions Locked

- **Model:** YOLOv8m (replacing DETR-ResNet50)
- **Input resolution:** 640×480
- **FPS:** 10, 300 frames/video, save every 2nd
- **Negative video ratio:** 25% of all videos (configured in `generator/pipeline.py`)
- **Kitchen target:** ~150 images (49 upscaled + ~100 new Gemini)
- **Train/val/test split:** Kitchen-level — same background NEVER appears in two splits
- **Held-out kitchens:** 10% never rendered in main batch (used for final test eval)
- **Training compute:** Google Colab Pro T4
- **Dataset storage:** Google Drive 2TB
- **Depth cache:** Pre-computed locally on M3 Pro, uploaded once to Drive
- **Labeling tool (real data):** Label Studio (free, local)

---

## Open Questions

| # | Question | Impact |
|---|----------|--------|
| A | Camera resolution and angle at NC State cafeteria? | Tune simulation params |
| B | Color or IR mode at time of recording? | Adjust grayscale ratio (currently 30%) |
| C | All 3 pest types in test, or subset? | Adjust class weights |
| D | Will real video be available before final test? | Determines if Step 9 happens |
