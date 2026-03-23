#!/bin/bash
#SBATCH --job-name=pest-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ── Usage ──────────────────────────────────────────────────────────────────────
# sbatch evaluate.sbatch /path/to/model/ /path/to/dataset/ [split] [threshold]
#
# Args:
#   $1 = path to fine-tuned DETR model
#   $2 = path to COCO dataset
#   $3 = split to evaluate (default: test)
#   $4 = detection threshold (default: 0.5)
# ───────────────────────────────────────────────────────────────────────────────

MODEL_PATH=${1:?"Usage: sbatch evaluate.sbatch <model_path> <data_dir> [split] [threshold]"}
DATA_DIR=${2:?"Usage: sbatch evaluate.sbatch <model_path> <data_dir> [split] [threshold]"}
SPLIT=${3:-"test"}
THRESHOLD=${4:-"0.5"}

mkdir -p logs

echo "=== Pest Detection Evaluation ==="
echo "Model:     $MODEL_PATH"
echo "Data:      $DATA_DIR"
echo "Split:     $SPLIT"
echo "Threshold: $THRESHOLD"
echo "=================================="

module load Python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

cd "$(dirname "$0")/.."

python model/evaluate_detection.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --split "$SPLIT" \
    --threshold "$THRESHOLD" \
    --output_json "model/eval_${SPLIT}_report.json"

echo "=== DONE ==="
