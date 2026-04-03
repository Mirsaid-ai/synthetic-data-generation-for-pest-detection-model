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

---

## Compute & Storage Resources

| Resource | Availability | Role |
|----------|-------------|------|
| Local MacBook M3 Pro (MPS) | Available | Prototyping, upscaling, light runs |
| Google Colab Pro (T4, 16GB) | Available | Primary training + batch rendering |
| Google Drive 2TB | Available | Dataset storage (pipeline ↔ Colab) |
| Duke Cluster | ❌ No access | — |
| Real video (NC State) | Not yet — to be obtained | Fine-tuning (high priority when available) |

---

## Current Asset Inventory

| Asset | Count | Status |
|-------|-------|--------|
| Anubhav curated kitchen images | 70 | ⚠️ Only 256×256 — must upscale |
| Pipeline approved kitchen images | 5 JPG + 5 PNG | ✅ Higher resolution |
| Sagnik sprites (mouse/rat/cockroach) | 36 PNGs (12/type) | ✅ Ready |
| Anubhav generator code | `generator/` | ✅ Merged into pipeline branch |
| Pipeline video generator | `video_generator/` | ⚠️ Has bugs (see PROJECT_CONTEXT.md) |
| DETR training code | `model/` | ⚠️ Wrong model — replacing with YOLOv8 |
| Anubhav training code | `training/` | ✅ Present, needs review |

---

## Step-by-Step Plan

---

### Step 1 — Upscale Existing 70 Kitchen Images
**Where:** Local M3 Pro
**Time:** ~1–2 hours

Use Real-ESRGAN (4× upscale) to bring all 70 images from 256×256 → 1024×1024.
These images are already curated — free diversity gain with no extra generation cost.

**Output:** `generator/kitchen_img/curated_img/` — same files, higher resolution

**Tool:** `scripts/upscale_images.py` (to be written)
```
Input:   generator/kitchen_img/curated_img/*.jpg (256×256)
Output:  generator/kitchen_img/curated_img/*.jpg (1024×1024, overwrite)
Model:   RealESRGAN_x4plus
```

---

### Step 2 — Generate New Commercial Kitchen Images
**Where:** Local (Gemini API via `kitchen_image_gen/` app)
**Time:** ~2–3 hours (generate ~250, approve ~100)

**Target:** 100 new images, majority cafeteria/commercial kitchen style, overhead angle.

**Key prompts to use:**
- `"Commercial cafeteria kitchen floor, overhead CCTV camera angle, ceramic tile floor, fluorescent lighting, photorealistic, no people, empty"`
- `"University cafeteria kitchen, bird's eye view from ceiling security camera, stainless steel equipment, white tile floor, bright overhead lights, empty floor"`
- `"Restaurant kitchen floor from ceiling-mounted security camera, wide angle, industrial kitchen, no pests, clean tile floor"`
- `"Institutional kitchen, top-down view, commercial cooking equipment, sealed concrete floor, overhead fluorescent lighting"`
- `"Home kitchen floor, side angle from counter height, linoleum floor, natural window light"` (diversity)

**After approval:** Move to `generator/kitchen_img/curated_img/`

**Total after Steps 1+2:** ~170 kitchen images at ≥1024×1024

---

### Step 3 — Add CCTV Simulation to Video Compositor
**Where:** Local (code change to `generator/compositing.py`)
**Time:** ~2–3 hours

Since the test environment is a cafeteria security camera, simulate these conditions
**per-video** (not per-frame — real cameras are consistent within a clip):

| Effect | Setting | Coverage |
|--------|---------|----------|
| Gaussian noise | σ = random(5, 20) | 100% of videos |
| JPEG compression | quality = random(60, 80) | 100% of videos |
| Brightness variation | factor = random(0.7, 1.3) | 100% of videos |
| Grayscale / IR mode | — | 30% of videos |
| Motion blur on sprites | kernel ∝ pest speed | 100% of videos |
| Resolution halve+upsample | ÷2 then ×2 | 20% of videos |

Apply to the composited frame before writing to video.

---

### Step 4 — Fix Critical Bugs in Video Generator ✅ DONE
**Completed:** April 2026

On inspection, most bugs listed in `CODEBASE_ANALYSIS.md` were already fixed in the current
code. Only one real fix was needed:

| # | File | Status |
|----|------|--------|
| 1 | `add_pests_to_kitchen.py` line 558 | ✅ **Fixed** — removed `1.0 -` from `depth_at()` call |
| 2 | `add_pests_to_kitchen.py` `mask_to_rle()` | ✅ Already correct — pycocotools used as primary |
| 3 | `generate_depth_map.py` JPG input | ✅ Already correct — `Path.with_suffix("")` used |
| 4 | `generate_floor_mask.py` depth semantics | ✅ Already correct — uses `< thresh_norm` |

---

### Step 5 — Batch Render 3,400 Videos
**Where:** Google Colab CPU (5–6 parallel sessions) → Google Drive
**Time:** ~4–6 hours wall time

**Video configuration:**

| Parameter | Value |
|-----------|-------|
| Resolution | 640×480 |
| FPS | 10 |
| Duration | 20 seconds (200 frames) |
| Kitchens | 170 |
| Videos per kitchen | 20 |
| Total videos | **3,400** |

**Video mix (applied per kitchen):**

| Type | % | Count | Purpose |
|------|---|-------|---------|
| No pest (negative) | 25% | 850 | Train FPR control — required |
| 1 pest (random type) | 35% | 1,190 | Clean single-instance learning |
| 2–3 pests (mixed types) | 40% | 1,360 | Realistic scenario |

**Estimated output:**
- ~340,000 total frames (extract every 2nd → ~170,000 training frames)
- Storage: ~50–70 GB on Google Drive

**Colab setup:**
```python
from google.colab import drive
drive.mount('/content/drive')
# clone repo / upload code
# run: python generator/pipeline.py --image_dir /content/drive/MyDrive/kitchens/ \
#          --output_dir /content/drive/MyDrive/pest_videos/ --n 20
```

---

### Step 6 — Extract Frames + Convert to YOLOv8 Format
**Where:** Local or Colab
**Time:** ~1 hour

Run `video_generator/extract_frames.py` to extract frames from all videos, then
convert the COCO JSON annotations to YOLOv8 format (`.txt` label files + `data.yaml`).

**Output structure:**
```
pest_dataset/
├── images/
│   ├── train/   (~136,000 frames, 80%)
│   ├── val/     (~17,000 frames, 10%)
│   └── test/    (~17,000 frames, 10%)
├── labels/
│   ├── train/   (YOLOv8 .txt files)
│   ├── val/
│   └── test/
└── data.yaml    (class names: mouse, rat, cockroach)
```

**Hold-out:** 17 kitchens (10%) are never used in video generation — reserved for
final evaluation with fresh synthetic test data.

**Tool:** `scripts/coco_to_yolo.py` (to be written)

---

### Step 7 — Train YOLOv8m on Google Colab Pro
**Where:** Google Colab Pro (T4, 16GB)
**Time:** ~6–8 hours (background execution, Pro keeps session alive)

**Model:** `yolov8m.pt` (pretrained on COCO — includes animal classes)

**Training command:**
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

**Checkpoints saved to Google Drive** — safe against Colab disconnects.

---

### Step 8 — Threshold Calibration + Evaluation
**Where:** Local M3 Pro
**Time:** ~1–2 hours

**Test sets:**
1. **Synthetic test** — 17 held-out kitchens, fresh videos never seen in training
2. **Negative test** — 200 frames of empty kitchen (no pests) from held-out kitchens

**Process:**
1. Run inference on all test frames, collect raw confidence scores
2. Sweep confidence threshold 0.05 → 0.95 in steps of 0.05
3. At each threshold: compute recall and FPR
4. Find threshold T where: recall(T) ≥ 0.80 AND FPR(T) < 0.05
5. If no such T exists → return to Step 7 (more epochs or more negative data)

**Tool:** `scripts/calibrate_threshold.py` (to be written)

---

### Step 9 — Real Video Fine-Tuning (When Video Is Available)
**Where:** Google Colab Pro
**Time:** ~1–2 hours labeling + ~1 hour training

This step is the highest single-accuracy lever. Even 200–500 labeled real frames
can close the remaining sim-to-real gap and push the model comfortably above target.

**Labeling tool:** [Label Studio](https://labelstud.io) — free, runs locally, exports YOLO format directly.

**Process:**
1. Install Label Studio locally: `pip install label-studio`
2. Load real cafeteria video frames
3. Draw bounding boxes, label class (mouse / rat / cockroach)
4. Target: **300–500 labeled frames** — 2–3 hours of work
5. Fine-tune YOLOv8m from Step 7 checkpoint:
   ```bash
   yolo train model=/path/to/best.pt data=real_data.yaml \
     epochs=30 lr0=0.0001 batch=8
   ```
6. Re-run threshold calibration (Step 8) on real test frames

---

## Summary Timeline

| Step | Task | Where | Est. Time |
|------|------|--------|-----------|
| 1 | Upscale 70 images (Real-ESRGAN) | Local | 1–2 hrs |
| 2 | Generate 100 new kitchen images | Local (Gemini) | 2–3 hrs |
| 3 | Add CCTV simulation to compositor | Local (code) | 2–3 hrs |
| 4 | Fix 3 critical bugs | Local (code) | 1–2 hrs |
| 5 | Render 3,400 videos | Colab CPU | 4–6 hrs |
| 6 | Extract frames + convert to YOLO | Local/Colab | 1 hr |
| 7 | Train YOLOv8m | Colab Pro T4 | 6–8 hrs |
| 8 | Threshold calibration + eval | Local | 1–2 hrs |
| 9 | Fine-tune on real frames (when available) | Colab Pro | 3–4 hrs |
| **Total** | | | **~2–3 days** |

---

## Scripts To Be Written

| Script | Purpose | Step |
|--------|---------|------|
| `scripts/upscale_images.py` | Real-ESRGAN 4× upscale of kitchen images | 1 |
| `scripts/coco_to_yolo.py` | Convert COCO JSON annotations → YOLOv8 txt format | 6 |
| `scripts/calibrate_threshold.py` | Sweep threshold, plot recall/FPR curve, find optimal T | 8 |
| `scripts/render_batch_colab.py` | Drive-aware batch rendering script for Colab | 5 |

---

## Key Decisions Locked

- **Model:** YOLOv8m (replacing DETR-ResNet50)
- **Input resolution:** 640×480
- **FPS:** 10
- **Negative video ratio:** 25% of all videos
- **Kitchen target:** 170 images (70 upscaled + 100 new)
- **Training frames:** ~170,000
- **Training compute:** Google Colab Pro T4
- **Dataset storage:** Google Drive 2TB
- **Labeling tool (real data):** Label Studio (free, local)
- **Labeling target:** 300–500 real frames when video available

---

## Open Questions

| # | Question | Impact |
|---|----------|--------|
| A | Camera resolution and angle at NC State cafeteria? | Tune simulation params |
| B | Color or IR mode at time of recording? | Adjust grayscale ratio |
| C | All 3 pest types in test, or subset? | Adjust class weights |
| D | Will real video be available before final test? | Determines if Step 9 happens |
