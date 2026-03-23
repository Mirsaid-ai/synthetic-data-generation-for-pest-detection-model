#!/bin/bash
#SBATCH --job-name=pest-detr-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ── Usage ──────────────────────────────────────────────────────────────────────
# sbatch train_model.sbatch /path/to/merged_dataset/ exp1_head
#
# Args:
#   $1 = path to merged COCO dataset
#   $2 = experiment name (default: experiment)
#   $3 = strategy: head|partial|full (default: head)
# ───────────────────────────────────────────────────────────────────────────────

DATA_DIR=${1:?"Usage: sbatch train_model.sbatch <data_dir> [exp_name] [strategy]"}
EXP_NAME=${2:-"experiment"}
STRATEGY=${3:-"head"}

mkdir -p logs

echo "=== DETR Training ==="
echo "Data dir:   $DATA_DIR"
echo "Experiment: $EXP_NAME"
echo "Strategy:   $STRATEGY"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "====================="

module load Python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

cd "$(dirname "$0")/.."

STRATEGY_FLAGS=""
case "$STRATEGY" in
    head)    STRATEGY_FLAGS="--freeze_backbone --epochs 20" ;;
    partial) STRATEGY_FLAGS="--partial_freeze 2 --epochs 30" ;;
    full)    STRATEGY_FLAGS="--epochs 50" ;;
    *)       echo "Unknown strategy: $STRATEGY (use head|partial|full)"; exit 1 ;;
esac

python model/finetune_detr.py \
    --data_dir "$DATA_DIR" \
    --output_dir "./model/detr_finetuned_${EXP_NAME}" \
    --experiment_name "$EXP_NAME" \
    --batch_size 4 \
    --augment \
    $STRATEGY_FLAGS

echo "=== DONE ==="
