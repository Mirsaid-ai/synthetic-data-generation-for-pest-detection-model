# Pipeline Usage Guide

## Quick Start

### 1. Generate Kitchen Images (Web UI)

```bash
cd kitchen_image_gen
cp .env.example .env.local
# Edit .env.local and add your GEMINI_API_KEY

npm install
npm run dev
# Open http://localhost:3000
```

- Enter prompts describing diverse kitchens
- Generate, review, modify, and approve images
- Approved images are saved to `kitchen_image_gen/public/approved_images/`

### 2. Generate Synthetic Pest Videos

**Single image (local):**
```bash
cd video_generator
python run_pipeline.py --image ../kitchen_image_gen/public/approved_images/kitchen_abc123.png \
                       --output_dir pipeline_out/ --n 30 --jobs 4
```

**All approved images (local):**
```bash
cd video_generator
python run_batch_pipeline.py \
    --image_dir ../kitchen_image_gen/public/approved_images/ \
    --output_dir pipeline_out/ \
    --n 30 --jobs 4 --every_n 3 --merge
```

**On Duke cluster:**
```bash
sbatch slurm/generate_videos.sbatch ./kitchen_images/ ./pipeline_out/ 30
```

### 3. Train DETR Model

**Local:**
```bash
cd model
python finetune_detr.py \
    --data_dir ../video_generator/pipeline_out/merged_dataset \
    --freeze_backbone --epochs 20 \
    --experiment_name exp1_head_only
```

**On Duke cluster:**
```bash
# Head-only (safest, start here)
sbatch slurm/train_model.sbatch ./video_generator/pipeline_out/merged_dataset exp1_head head

# Partial fine-tune (better accuracy)
sbatch slurm/train_model.sbatch ./video_generator/pipeline_out/merged_dataset exp2_partial partial

# Full fine-tune (best with lots of data)
sbatch slurm/train_model.sbatch ./video_generator/pipeline_out/merged_dataset exp3_full full
```

### 4. Evaluate

```bash
cd model
python evaluate_detection.py \
    --model_path ./detr_finetuned \
    --data_dir ../video_generator/pipeline_out/merged_dataset \
    --split test --threshold 0.5 \
    --output_json eval_report.json
```

**On Duke cluster:**
```bash
sbatch slurm/evaluate.sbatch ./model/detr_finetuned ./video_generator/pipeline_out/merged_dataset
```

### 5. Run Inference on New Video

```bash
cd model
python inference_detection.py \
    --model_path ./detr_finetuned \
    --video ../test_video.mp4 \
    --threshold 0.5 \
    --visualize --vis_dir ./visualizations/
```

## Recommended Dataset Scale

| Parameter | Recommended | Minimum |
|-----------|------------|---------|
| Kitchen images | 30-50 | 10 |
| Videos per kitchen | 30-50 | 10 |
| Total videos | 1000-2500 | 100 |
| Frame extraction | Every 3rd frame | Every 5th |
| Storage needed | 100-300 GB | 20 GB |

## Project Structure

```
synthetic_video_gen/
├── kitchen_image_gen/     # Next.js web UI for Gemini kitchen image generation
│   ├── app/               # Pages and API routes
│   ├── components/        # React components
│   ├── lib/               # Gemini API wrapper, storage utilities
│   └── public/approved_images/  # Approved kitchen images
│
├── video_generator/       # Synthetic pest video generation pipeline
│   ├── run_pipeline.py            # Single-image pipeline orchestrator
│   ├── run_batch_pipeline.py      # Batch processing for all images
│   ├── generate_depth_map.py      # MiDaS depth estimation
│   ├── generate_floor_mask.py     # SegFormer floor segmentation
│   ├── generate_configs.py        # Random pest config generation
│   ├── add_pests_to_kitchen.py    # Core renderer + COCO annotations
│   ├── batch_render.py            # Parallel video rendering
│   ├── extract_frames.py          # Frame extraction + dataset assembly
│   └── merge_datasets.py          # Merge per-image datasets into one
│
├── model/                 # DETR object detection training
│   ├── finetune_detr.py           # Fine-tune DETR on pest dataset
│   ├── inference_detection.py     # Run detection on images/video
│   ├── evaluate_detection.py      # Compute mAP, detection rate, FPR
│   └── requirements.txt
│
└── slurm/                 # Duke cluster job templates
    ├── generate_videos.sbatch
    ├── train_model.sbatch
    └── evaluate.sbatch
```
