#!/bin/bash
#SBATCH --job-name=pest-video-gen
#SBATCH --output=logs/video_gen_%j.out
#SBATCH --error=logs/video_gen_%j.err
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# ── Usage ──────────────────────────────────────────────────────────────────────
# sbatch generate_videos.sbatch /path/to/kitchen_images/ /path/to/output/ 30
#
# Args:
#   $1 = directory of approved kitchen images
#   $2 = output directory (default: pipeline_out/)
#   $3 = number of videos per image (default: 30)
# ───────────────────────────────────────────────────────────────────────────────

IMAGE_DIR=${1:?"Usage: sbatch generate_videos.sbatch <image_dir> [output_dir] [n_videos]"}
OUTPUT_DIR=${2:-"pipeline_out"}
N_VIDEOS=${3:-30}

mkdir -p logs

echo "=== Pest Video Generation ==="
echo "Image dir:  $IMAGE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Videos/img: $N_VIDEOS"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================="

module load Python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

cd "$(dirname "$0")/.."

python video_generator/run_batch_pipeline.py \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n "$N_VIDEOS" \
    --jobs 4 \
    --every_n 3 \
    --skip_existing \
    --merge

echo "=== DONE ==="
