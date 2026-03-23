"""
merge_datasets.py
=================
Merges multiple per-kitchen COCO datasets into one unified dataset.

Each run of run_pipeline.py produces a dataset under:
    out/<image_stem>/dataset/
        images/{train,val,test}/
        annotations/{train,val,test}.json

This script takes a root directory containing multiple such dataset dirs,
merges all splits into a single unified COCO dataset with re-indexed IDs.

Usage:
    python merge_datasets.py \
        --input_dir  pipeline_out/ \
        --output_dir pipeline_out/merged_dataset/

    # Custom split rebalance (pool all frames, re-split)
    python merge_datasets.py \
        --input_dir  pipeline_out/ \
        --output_dir pipeline_out/merged_dataset/ \
        --resplit --split 0.8 0.1 0.1 --seed 42
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "cockroach", "supercategory": "pest"},
    {"id": 3, "name": "rat",       "supercategory": "pest"},
]


def find_dataset_dirs(input_dir):
    """Find all subdirectories that contain annotations/ and images/ folders."""
    datasets = []
    root = Path(input_dir)
    for candidate in sorted(root.rglob("annotations")):
        dataset_dir = candidate.parent
        if (dataset_dir / "images").is_dir():
            datasets.append(dataset_dir)
    return datasets


def load_and_reindex(ann_path, image_base_dir, img_id_offset, ann_id_offset):
    """Load a COCO JSON, reindex IDs, and resolve image paths."""
    with open(ann_path) as f:
        coco = json.load(f)

    old_to_new_img = {}
    new_images = []
    for img in coco.get("images", []):
        old_id = img["id"]
        new_id = old_id + img_id_offset
        old_to_new_img[old_id] = new_id
        img["id"] = new_id
        new_images.append(img)

    new_anns = []
    for ann in coco.get("annotations", []):
        ann["id"] = ann["id"] + ann_id_offset
        ann["image_id"] = old_to_new_img.get(ann["image_id"], ann["image_id"])
        new_anns.append(ann)

    max_img_id = max((img["id"] for img in new_images), default=img_id_offset)
    max_ann_id = max((ann["id"] for ann in new_anns), default=ann_id_offset)

    return new_images, new_anns, max_img_id, max_ann_id


def merge_split(dataset_dirs, split_name, output_dir, img_id_offset=0, ann_id_offset=0):
    """Merge one split (train/val/test) across all dataset dirs."""
    all_images = []
    all_anns = []
    copied = 0

    out_img_dir = Path(output_dir) / "images" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for ds_dir in dataset_dirs:
        ann_path = ds_dir / "annotations" / f"{split_name}.json"
        if not ann_path.exists():
            continue

        images, anns, max_img, max_ann = load_and_reindex(
            ann_path, ds_dir, img_id_offset, ann_id_offset
        )

        for img in images:
            old_file = img["file_name"]
            base_name = os.path.basename(old_file)

            # Try multiple source paths
            src = None
            for candidate in [
                ds_dir / old_file,
                ds_dir / "images" / split_name / base_name,
                ds_dir / base_name,
            ]:
                if candidate.exists():
                    src = candidate
                    break

            if src:
                dst = out_img_dir / base_name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                img["file_name"] = f"images/{split_name}/{base_name}"

        all_images.extend(images)
        all_anns.extend(anns)

        img_id_offset = max_img + 1
        ann_id_offset = max_ann + 1

    if not all_images:
        return 0, 0, img_id_offset, ann_id_offset

    coco_merged = {
        "info": {
            "description": f"Merged synthetic pest dataset - {split_name}",
            "version": "1.0",
        },
        "categories": CATEGORIES,
        "images": all_images,
        "annotations": all_anns,
    }

    ann_out_dir = Path(output_dir) / "annotations"
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_out_dir / f"{split_name}.json", "w") as f:
        json.dump(coco_merged, f)

    print(f"  {split_name:5s}: {len(all_images):6d} images, "
          f"{len(all_anns):7d} annotations, {copied} files copied")

    return len(all_images), len(all_anns), img_id_offset, ann_id_offset


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO pest datasets into one.")
    parser.add_argument("--input_dir", required=True,
                        help="Root dir containing per-kitchen pipeline outputs")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for merged dataset")
    parser.add_argument("--resplit", action="store_true",
                        help="Pool all frames and re-split instead of merging per-split")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = find_dataset_dirs(args.input_dir)
    if not datasets:
        print(f"[ERROR] No datasets found under: {args.input_dir}")
        print("  Expected structure: <input_dir>/<name>/dataset/annotations/")
        return

    print(f"Found {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  {d}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nMerging into: {args.output_dir}")

    total_images = 0
    total_anns = 0
    img_offset = 0
    ann_offset = 0

    for split in ["train", "val", "test"]:
        n_imgs, n_anns, img_offset, ann_offset = merge_split(
            datasets, split, args.output_dir, img_offset, ann_offset
        )
        total_images += n_imgs
        total_anns += n_anns

    # Write dataset info
    info = {
        "source_datasets": [str(d) for d in datasets],
        "total_images": total_images,
        "total_annotations": total_anns,
        "categories": CATEGORIES,
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[DONE] Merged dataset: {args.output_dir}")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_anns}")


if __name__ == "__main__":
    main()
