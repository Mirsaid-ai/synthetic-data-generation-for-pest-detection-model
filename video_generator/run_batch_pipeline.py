"""
run_batch_pipeline.py
=====================
Process ALL approved kitchen images through the full video generation pipeline.

Takes a directory of kitchen images and runs run_pipeline.py for each one,
producing per-image datasets, then optionally merges them.

Usage:
    # Generate videos for all images in a directory
    python run_batch_pipeline.py \
        --image_dir ../kitchen_image_gen/public/approved_images/ \
        --output_dir pipeline_out/ \
        --n 30 --jobs 4

    # Also merge into a unified dataset
    python run_batch_pipeline.py \
        --image_dir ../kitchen_image_gen/public/approved_images/ \
        --output_dir pipeline_out/ \
        --n 30 --merge

    # Skip already-processed images
    python run_batch_pipeline.py \
        --image_dir ./kitchen_images/ \
        --output_dir pipeline_out/ \
        --n 30 --skip_existing
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(HERE, "run_pipeline.py")
MERGE_SCRIPT = os.path.join(HERE, "merge_datasets.py")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(image_dir):
    img_dir = Path(image_dir)
    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Batch process kitchen images through the full pipeline.")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing approved kitchen images")
    parser.add_argument("--output_dir", default="pipeline_out",
                        help="Root output directory")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of videos per kitchen image (default: 30)")
    parser.add_argument("--mice", type=int, nargs=2, default=[0, 3])
    parser.add_argument("--cockroaches", type=int, nargs=2, default=[0, 5])
    parser.add_argument("--rats", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--duration", type=float, nargs=2, default=[15, 30])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel render jobs per image")
    parser.add_argument("--every_n", type=int, default=3,
                        help="Extract every Nth frame (default: 3)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have output dirs")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all per-image datasets into one after processing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = find_images(args.image_dir)
    if not images:
        print(f"[ERROR] No images found in: {args.image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} kitchen image(s)")
    print(f"Videos per image: {args.n}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    succeeded = 0
    failed = 0

    for idx, img_path in enumerate(images):
        stem = img_path.stem
        out_dir = os.path.join(args.output_dir, stem)

        if args.skip_existing and os.path.isdir(out_dir):
            dataset_dir = os.path.join(out_dir, "dataset")
            if os.path.isdir(dataset_dir):
                print(f"[{idx+1}/{len(images)}] SKIP {img_path.name} (exists)")
                succeeded += 1
                continue

        print(f"\n[{idx+1}/{len(images)}] Processing: {img_path.name}")

        cmd = [
            sys.executable, PIPELINE_SCRIPT,
            "--image", str(img_path),
            "--output_dir", args.output_dir,
            "--n", str(args.n),
            "--mice", str(args.mice[0]), str(args.mice[1]),
            "--cockroaches", str(args.cockroaches[0]), str(args.cockroaches[1]),
            "--rats", str(args.rats[0]), str(args.rats[1]),
            "--duration", str(args.duration[0]), str(args.duration[1]),
            "--fps", str(args.fps),
            "--jobs", str(args.jobs),
            "--every_n", str(args.every_n),
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed + idx)]

        result = subprocess.run(cmd)
        if result.returncode == 0:
            succeeded += 1
        else:
            failed += 1
            print(f"  [ERROR] Pipeline failed for {img_path.name}")

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {succeeded} succeeded, {failed} failed, {len(images)} total")

    if args.merge and succeeded > 0:
        print(f"\nMerging datasets...")
        merged_dir = os.path.join(args.output_dir, "merged_dataset")
        merge_cmd = [
            sys.executable, MERGE_SCRIPT,
            "--input_dir", args.output_dir,
            "--output_dir", merged_dir,
        ]
        result = subprocess.run(merge_cmd)
        if result.returncode == 0:
            print(f"Merged dataset: {merged_dir}")
        else:
            print("[ERROR] Merge failed")


if __name__ == "__main__":
    main()
