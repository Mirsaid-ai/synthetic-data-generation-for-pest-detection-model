# Project Context — Synthetic Pest Video Generation
> Last updated: April 2026
> Branch: `pipeline`

---

## 1. What This Project Is

End-to-end pipeline that generates **fully synthetic, annotated video datasets** for training pest-detection models (mice, cockroaches, rats in kitchens). The goal is to avoid manual data collection and labeling.

**Final test target:** Real kitchen video footage (camera type TBD — not confirmed CCTV) — model must achieve:
- **≥80% true detection rate** (recall)
- **<5% false positive rate** (FPR)
- Measured per-frame on test video data

---

## 2. Repository Structure

```
synthetic_video_gen/
├── kitchen_image_gen/          # Next.js + Gemini API web UI for kitchen image generation
├── video_generator/            # Core Python pipeline (this is the main focus)
├── model/                      # DETR fine-tuning, inference, evaluation
├── slurm/                      # Duke cluster SBATCH job templates (GPU)
├── README.md
├── CODEBASE_ANALYSIS.md        # Detailed bug list (generated March 2026)
├── VideoGeneratorUsage.md
└── PROJECT_CONTEXT.md          # ← this file
```

---

## 3. Full Pipeline Flow

```
kitchen.png
  ↓
[Step 1] generate_depth_map.py
  → MiDaS DPT_Large (torch.hub) — grayscale depth map, min-max normalized
  ↓
[Step 2] generate_floor_mask.py
  → SegFormer-B2 (ADE20K, HuggingFace) — binary floor mask
  → Post-process: fill holes → speck removal → Gaussian smooth
  ↓
[Step 3] generate_configs.py
  → Random JSON configs: pest types/counts/sizes/speeds
  ↓
[Step 4] batch_render.py → add_pests_to_kitchen.py (parallel)
  → Procedural pest sprites composited onto background
  → Per-frame COCO annotations (bbox + polygon segmentation + track_id)
  → Output: videos/*.mp4 + videos/*_coco.json
  ↓
[Step 5] extract_frames.py
  → JPEG frames + train/val/test COCO dataset split
  → Output: dataset/images/{train,val,test}/ + dataset/annotations/*.json

[Training] model/finetune_detr.py
  → facebook/detr-resnet-50, freeze strategies: head-only / partial / full
  → Augmentation: ColorJitter, GaussianBlur, RandomGrayscale, Sharpness

[Inference] model/inference_detection.py
[Evaluation] model/evaluate_detection.py
```

---

## 4. Bug Status in `pipeline` Branch

### ✅ Fixed
| # | File | Notes |
|---|------|-------|
| 1 | `add_pests_to_kitchen.py` line 558 | **Depth inversion fixed** — removed `1.0 -` from `depth_at()` call; perspective scaling now correct |
| 2 | `generate_depth_map.py` | JPG input was already safe — uses `Path.with_suffix("")` |
| 3 | `generate_depth_map.py` | `if __name__` guard already present |
| 4 | `generate_floor_mask.py` | Depth refinement already uses `< thresh_norm` (correct direction) |
| 5 | `add_pests_to_kitchen.py` | `mask_to_rle()` already uses pycocotools as primary path |

### 🟠 Still Open (lower priority — do not block pipeline)
| # | File | Issue |
|---|------|-------|
| 6 | `batch_render.py` | `capture_output=True` swallows all render stdout/stderr |
| 7 | `generate_configs.py` | `generate_config()` dead stub always returns `None` |
| 8 | `extract_frames.py` | `old_id` computed but never used |
| 9 | `extract_frames.py` | `img_rec["file_name"]` mutated in-place on shared dict |
| 10 | All scripts | `floor_thresh` vs `depth_thresh` naming inconsistency |

---

## 5. Branch Comparison: `pipeline` vs `upstream/anubhav_v1`

### Architecture
| Aspect | pipeline | anubhav_v1 |
|--------|----------|-----------|
| Organization | Single 677-line monolithic file | 5 modular files (pest_animation, pest_models, compositing, pipeline, depth_estimator) |
| Sprite system | Procedural OpenCV (circles/lines) | PIL + pre-rendered Sagnik sprites (36 PNGs, 12 per pest type) |
| Depth analysis | MiDaS only | MiDaS + surface normals + gravity estimation + per-surface group masks |
| Movement | Flat binary floor mask, steering angle | Surface-aware (up/down/side_*), depth-aware speed cap, surface stickiness param |
| COCO output | bbox + polygon segmentation + track_id | bbox only (no segmentation, no track_id) |
| Contact shadow | Yes (ellipse) | No |
| Web app | No | Yes (Flask, gallery UI) |
| Frame extraction | Yes (extract_frames.py) | No |

### Where anubhav_v1 is Better
1. **Sprite quality** — Sagnik pre-rendered assets vs. procedural circles; rats recolored from mouse sprites via `ImageOps.colorize`
2. **Scene understanding** — `depth_estimator.py` (917 lines) computes surface normals, gravity direction, per-surface masks
3. **Movement realism** — surface stickiness (0.97), per-surface probability maps, depth-aware speed capping
4. **Architecture** — modular, easier to maintain and extend
5. **Avoids the depth inversion bug** — uses implicit constraints instead of explicit depth scaling

### Where `pipeline` is Better
1. **Annotation completeness** — instance segmentation masks (polygon + RLE fallback) + `track_id` across frames
2. **Contact shadows** — ellipse shadow under pests
3. **Simpler standalone CLI** — no web app dependency
4. **Frame extraction tooling** — `extract_frames.py` handles train/val/test splitting

### Merge Decision
**Bring from anubhav_v1 into pipeline:**
- Sprite system (Sagnik assets + PIL compositing)
- `depth_estimator.py` for surface-aware scene analysis
- Modular file structure

**Keep from pipeline:**
- Instance segmentation masks
- Track IDs
- Contact shadows
- `extract_frames.py`, `merge_datasets.py`

---

## 6. Sim-to-Real Gap Assessment

### Current Rendering Gaps vs. Real CCTV

| Property | Current Synthetic | Real CCTV | Impact |
|----------|------------------|-----------|--------|
| Color mode | Full RGB | **IR grayscale at night** (pest cameras are nocturnal) | 🔴 Critical |
| Noise | None | Heavy sensor noise in shadows/edges | 🟠 High |
| Compression | mp4v (clean) | H.264 at 500–2000kbps, visible blocking | 🟠 High |
| Motion blur | None | Present on fast-moving pests | 🟠 High |
| Camera angle | Eye-level kitchen photos | **Ceiling-mounted, wide-angle, overhead** | 🔴 Critical |
| Lens | None | Barrel distortion from wide-angle | 🟡 Medium |
| Occlusion | Pests always fully visible | Pests behind/under furniture | 🟡 Medium |
| Sprite realism | Procedural circles | Real fur/scale texture | 🟠 High |
| Lighting reaction | Flat sprite colors | Real ambient light, shadow, specular | 🟡 Medium |
| Background diversity | 12 images (current) | Highly variable | 🔴 Critical |

### Realistic Performance Estimates

Target: ≥80% recall AND <5% FPR simultaneously.

| Scenario | Recall | FPR | Passes? |
|----------|--------|-----|---------|
| Current pipeline, DETR, no changes | 35–50% | ~15–25% | No |
| + Merge anubhav_v1 sprites | 45–60% | ~10–20% | No |
| + Real domain augmentation (noise, blur, compression) | 60–72% | ~8–12% | No |
| + Better model (YOLOv8) + 70+ kitchens + threshold tuning | **72–80%** | **~5–8%** | Borderline |
| + Negative frames training + calibration | **78–84%** | **<5%** | **Likely yes** |
| + Any real labeled video frames (200+) | **82–90%** | **<4%** | **Yes** |

**Critical note:** The <5% FPR constraint is the harder one — it's easy to get 80% recall by lowering the confidence threshold, but that blows up FPR. Both must hold simultaneously.

---

## 7. Roadmap to 85%+ on Real CCTV

### Phase 1 — Fix & Merge (Priority)
- [ ] Fix depth inversion bug (`add_pests_to_kitchen.py` line 558: remove `1.0 -`)
- [ ] Fix RLE encoding (use `pycocotools.mask.encode()`)
- [ ] Fix JPG input overwrite in `generate_depth_map.py`
- [ ] Merge anubhav_v1 sprite system + depth_estimator into pipeline
- [ ] Switch video output codec to H.264

### Phase 2 — Real Video Simulation Layer
Camera type is TBD (not confirmed CCTV). Simulate a broad range of conditions:
- [ ] **Sensor noise injection** — Gaussian σ=5–15 (applies to all camera types)
- [ ] **H.264/JPEG compression simulation** — encode/decode at quality 55–75
- [ ] **Motion blur on sprites** — directional blur proportional to speed vector
- [ ] **Lighting variation** — random brightness/gamma per-video segment
- [ ] **Resolution downscale** — simulate lower-res capture
- [ ] **Random grayscale** (30–40% of videos) — covers IR night mode if camera turns out to be CCTV
- [ ] **Barrel distortion** (optional) — covers wide-angle lenses

### Phase 3 — Data Scale-Up
- [ ] Generate 100+ kitchen images with ceiling/overhead CCTV perspective via Gemini prompts
- [ ] Run batch: 100 kitchens × 30 configs = 3,000 videos
- [ ] Extract every 3rd frame → ~150K training frames target

### Phase 4 — Model Switch + Threshold Calibration
- [ ] Replace `facebook/detr-resnet-50` with **YOLOv8m** or **RT-DETR-L**
- [ ] Upgrade training augmentations:
  - RandomGrayscale p=0.3–0.4
  - MotionBlur kernel=3–7
  - JPEG compression q=50–80
  - Mosaic (YOLOv8 built-in, critical for small objects)
  - Perspective warp scale=0.3
- [ ] **Threshold calibration on held-out val set** — tune confidence threshold to jointly satisfy ≥80% recall AND <5% FPR (they trade off against each other)
- [ ] **Train with negative frames** (pest-free kitchen frames) — critical for FPR control

### Phase 5 — Real Data (Game Changer)
- [ ] Collect any real CCTV kitchen footage
- [ ] Label 200–500 frames manually
- [ ] Fine-tune on real frames after synthetic pretraining
- Expected result: **85–92% recall**

---

## 8. Key Files Quick Reference

| File | Lines | Role |
|------|-------|------|
| `video_generator/run_pipeline.py` | 307 | End-to-end orchestrator |
| `video_generator/add_pests_to_kitchen.py` | 677 | Core renderer + COCO annotations |
| `video_generator/generate_depth_map.py` | 68 | MiDaS depth estimation |
| `video_generator/generate_floor_mask.py` | 248 | SegFormer floor segmentation |
| `video_generator/generate_configs.py` | 153 | Random config generation |
| `video_generator/batch_render.py` | 147 | Parallel render launcher |
| `video_generator/extract_frames.py` | 337 | Frame extraction + dataset split |
| `video_generator/merge_datasets.py` | — | Merge per-image datasets |
| `model/finetune_detr.py` | 342 | DETR fine-tuning |
| `model/inference_detection.py` | — | Run detection on image/video |
| `model/evaluate_detection.py` | — | mAP, recall, FPR metrics |

---

## 9. Current Dataset State

- **Kitchen images:** 12 (5 JPG real photos + 5 Gemini-generated PNGs + metadata)
- **Location:** `kitchen_image_gen/public/approved_images/`
- **Minimum needed:** 50+ (target: 100+) overhead/CCTV-angle images

---

## 10. Cluster (Duke)

SBATCH templates in `slurm/`:
- `generate_videos.sbatch` — 1 GPU, 8 CPU, 32GB, 8h limit
- `train_model.sbatch` — 1 GPU, 4 CPU, 32GB, 6h limit
- `evaluate.sbatch` — same specs

Activate environment before jobs: `conda activate synthetic_pest` (or equivalent).
