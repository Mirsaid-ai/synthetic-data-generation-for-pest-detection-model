#!/bin/bash
#SBATCH --job-name=pest-yolov8-train
#SBATCH --output=logs/yolov8_%j.out
#SBATCH --error=logs/yolov8_%j.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# ── Usage ──────────────────────────────────────────────────────────────────────
# sbatch slurm/train_yolov8.sbatch /path/to/pest_dataset/ [exp_name] [epochs] [model]
#
# Args:
#   $1 = path to pest_dataset/ directory (must contain data.yaml)
#   $2 = experiment name (default: yolov8s_baseline)
#   $3 = number of epochs (default: 100)
#   $4 = model checkpoint (default: yolov8s.pt — use yolov8m.pt for larger model)
#
# Examples:
#   sbatch slurm/train_yolov8.sbatch ~/pest_project/pest_dataset yolov8s_baseline 100 yolov8s.pt
#   sbatch slurm/train_yolov8.sbatch ~/pest_project/pest_dataset yolov8m_v1 100 yolov8m.pt
# ───────────────────────────────────────────────────────────────────────────────

DATA_DIR=${1:?"Usage: sbatch slurm/train_yolov8.sbatch <data_dir> [exp_name] [epochs] [model]"}
EXP_NAME=${2:-"yolov8s_baseline"}
EPOCHS=${3:-100}
MODEL=${4:-"yolov8s.pt"}

DATA_YAML="${DATA_DIR%/}/data.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: data.yaml not found at $DATA_YAML"
    echo "Run scripts/build_dataset.py first."
    exit 1
fi

mkdir -p logs

echo "=== YOLOv8 Training ==="
echo "data.yaml:  $DATA_YAML"
echo "Model:      $MODEL"
echo "Experiment: $EXP_NAME"
echo "Epochs:     $EPOCHS"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================"

module load Python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || conda activate synthetic_pest 2>/dev/null || true

pip install ultralytics --quiet

cd "$(dirname "$0")/.."

yolo train \
    data="$DATA_YAML" \
    model="$MODEL" \
    name="$EXP_NAME" \
    project=runs/pest \
    epochs="$EPOCHS" \
    imgsz=640 \
    batch=-1 \
    device=0 \
    workers=8 \
    patience=20 \
    save=True \
    save_period=10 \
    augment=True \
    mosaic=1.0 \
    degrees=10 \
    fliplr=0.5 \
    hsv_h=0.015 \
    hsv_s=0.5 \
    hsv_v=0.4 \
    translate=0.1 \
    scale=0.3 \
    erasing=0.3

echo "=== DONE ==="
echo "Best weights: runs/pest/${EXP_NAME}/weights/best.pt"
