"""
extract_frames.py
=================
Extracts frames from rendered pest videos and assembles a single merged
COCO JSON dataset ready for training.

For each video it finds the matching *_coco.json sidecar, extracts frames
as JPEGs, and merges all annotations into one dataset JSON.

Usage:
    # Extract frames from all videos in a directory:
    python extract_frames.py \
        --video_dir  videos/ \
        --output_dir dataset/ \
        --split      0.8 0.1 0.1

    # Single video:
    python extract_frames.py \
        --video      output.mp4 \
        --output_dir dataset/

Output structure:
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── video_0000_frame_000000.jpg
    │   │   └── ...
    │   ├── val/
    │   └── test/
    ├── annotations/
    │   ├── train.json       ← full COCO JSON
    │   ├── val.json
    │   └── test.json
    └── dataset_info.json    ← summary stats

Arguments:
    --video_dir     Directory containing .mp4 + _coco.json pairs
    --video         Single video file (alternative to --video_dir)
    --output_dir    Root output directory (default: dataset/)
    --split         Train/val/test fractions (default: 0.8 0.1 0.1)
    --quality       JPEG quality 1-100 (default: 95)
    --every_n       Extract every Nth frame only (default: 1 = all frames)
    --no_empty      Skip frames with no pest annotations
    --seed          Random seed for train/val/test split
"""

import cv2
import json
import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "cockroach", "supercategory": "pest"},
]


# ─────────────────────────────────────────────
#  FRAME EXTRACTION
# ─────────────────────────────────────────────

def extract_frames_from_video(video_path, coco_data, output_dir,
                               quality=95, every_n=1, no_empty=False):
    """
    Extract JPEG frames from a video, returning list of
    (frame_record, [annotation_records]) for extracted frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    # Index annotations by frame (image_id)
    ann_by_frame = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        ann_by_frame[ann["image_id"]].append(ann)

    # Index images by frame_idx
    img_by_idx = {img["frame_idx"]: img for img in coco_data.get("images", [])}

    extracted = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            img_record = img_by_idx.get(frame_idx)
            if img_record is None:
                frame_idx += 1
                continue

            anns = ann_by_frame.get(frame_idx, [])

            if no_empty and len(anns) == 0:
                frame_idx += 1
                continue

            # Write JPEG
            fname = img_record["file_name"]
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            extracted.append((img_record, anns))

        frame_idx += 1

    cap.release()
    return extracted


# ─────────────────────────────────────────────
#  COCO MERGE + SPLIT
# ─────────────────────────────────────────────

def build_split_coco(all_records, split_name):
    """
    Build a COCO JSON dict from a list of (image_record, [ann_records]).
    Re-indexes image IDs and annotation IDs to be globally unique.
    """
    coco = {
        "info": {
            "description": f"Synthetic pest dataset — {split_name}",
            "version": "1.0",
        },
        "categories":  CATEGORIES,
        "images":      [],
        "annotations": [],
    }

    new_image_id = 1
    new_ann_id   = 1

    for img_record, anns in all_records:
        old_id = img_record["image_id"] if "image_id" in img_record else img_record["id"]
        new_img = {k: v for k, v in img_record.items()}
        new_img["id"] = new_image_id
        coco["images"].append(new_img)

        for ann in anns:
            new_ann = {k: v for k, v in ann.items()}
            new_ann["id"]       = new_ann_id
            new_ann["image_id"] = new_image_id
            coco["annotations"].append(new_ann)
            new_ann_id += 1

        new_image_id += 1

    return coco


def split_records(all_records, fracs, seed):
    """Split list into (train, val, test) by fractions."""
    rng = random.Random(seed)
    records = list(all_records)
    rng.shuffle(records)

    n     = len(records)
    n_tr  = int(n * fracs[0])
    n_val = int(n * fracs[1])

    train = records[:n_tr]
    val   = records[n_tr:n_tr+n_val]
    test  = records[n_tr+n_val:]
    return train, val, test


# ─────────────────────────────────────────────
#  DATASET INFO
# ─────────────────────────────────────────────

def compute_stats(coco, split_name):
    n_images = len(coco["images"])
    n_anns   = len(coco["annotations"])
    per_cat  = defaultdict(int)
    for ann in coco["annotations"]:
        per_cat[ann["category_id"]] += 1
    cat_names = {c["id"]: c["name"] for c in CATEGORIES}
    return {
        "split":       split_name,
        "images":      n_images,
        "annotations": n_anns,
        "per_category": {cat_names.get(k, str(k)): v for k, v in per_cat.items()},
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from pest videos and build COCO dataset.")
    parser.add_argument("--video_dir",  default=None,
                        help="Directory containing .mp4 + _coco.json pairs")
    parser.add_argument("--video",      default=None,
                        help="Single video file")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--split",      type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--quality",    type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--every_n",    type=int, default=1,
                        help="Extract every Nth frame (default: 1 = all)")
    parser.add_argument("--no_empty",   action="store_true",
                        help="Skip frames with no pest annotations")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Validate split
    if abs(sum(args.split) - 1.0) > 0.01:
        print(f"[ERROR] Split fractions must sum to 1.0 (got {sum(args.split):.2f})")
        sys.exit(1)

    # Collect video+annotation pairs
    pairs = []
    if args.video:
        video_path = Path(args.video)
        ann_path   = video_path.with_name(video_path.stem + "_coco.json")
        if not ann_path.exists():
            print(f"[ERROR] No annotation file found: {ann_path}")
            sys.exit(1)
        pairs.append((video_path, ann_path))
    elif args.video_dir:
        vdir = Path(args.video_dir)
        for mp4 in sorted(vdir.glob("*.mp4")):
            ann = mp4.with_name(mp4.stem + "_coco.json")
            if ann.exists():
                pairs.append((mp4, ann))
            else:
                print(f"  [WARN] No annotation for {mp4.name} — skipping")
    else:
        print("[ERROR] Provide --video or --video_dir"); sys.exit(1)

    if not pairs:
        print("[ERROR] No valid video+annotation pairs found"); sys.exit(1)

    print(f"[INFO] Found {len(pairs)} video(s)")

    # Output structure
    out       = Path(args.output_dir)
    img_dirs  = {s: out / "images" / s for s in ("train", "val", "test")}
    ann_dir   = out / "annotations"
    for d in list(img_dirs.values()) + [ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Temp dir for all extracted frames before splitting
    tmp_dir = out / "_tmp_frames"
    tmp_dir.mkdir(exist_ok=True)

    all_records = []   # list of (img_record, [ann_records])

    for v_idx, (video_path, ann_path) in enumerate(pairs):
        print(f"\n[{v_idx+1}/{len(pairs)}] {video_path.name}")
        with open(ann_path) as f:
            coco_data = json.load(f)

        records = extract_frames_from_video(
            video_path, coco_data, str(tmp_dir),
            quality=args.quality, every_n=args.every_n,
            no_empty=args.no_empty)

        print(f"  Extracted {len(records)} frames")
        all_records.extend(records)

    print(f"\n[INFO] Total frames extracted: {len(all_records)}")

    # Split
    train_r, val_r, test_r = split_records(all_records, args.split, args.seed)
    splits = [("train", train_r), ("val", val_r), ("test", test_r)]
    print(f"[INFO] Split: train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")

    all_stats = []

    for split_name, records in splits:
        if not records:
            print(f"  [WARN] {split_name} split is empty — skipping")
            continue

        # Move frames to correct split directory
        dest_img_dir = img_dirs[split_name]
        for img_rec, _ in records:
            src = tmp_dir / img_rec["file_name"]
            dst = dest_img_dir / img_rec["file_name"]
            if src.exists():
                shutil.move(str(src), str(dst))
            # Update file path in record to be relative to dataset root
            img_rec["file_name"] = f"images/{split_name}/{img_rec['file_name']}"

        # Build and save COCO JSON
        coco = build_split_coco(records, split_name)
        ann_path_out = ann_dir / f"{split_name}.json"
        with open(ann_path_out, "w") as f:
            json.dump(coco, f)

        stats = compute_stats(coco, split_name)
        all_stats.append(stats)

        print(f"  {split_name:5s}: {stats['images']:5d} images  "
              f"{stats['annotations']:6d} annotations  "
              f"{stats['per_category']}")

    # Clean up temp dir
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Dataset info summary
    info = {
        "total_videos":      len(pairs),
        "total_frames":      len(all_records),
        "every_n":           args.every_n,
        "no_empty_frames":   args.no_empty,
        "jpeg_quality":      args.quality,
        "split_fractions":   {"train": args.split[0],
                              "val":   args.split[1],
                              "test":  args.split[2]},
        "splits":            all_stats,
        "categories":        CATEGORIES,
    }
    info_path = out / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[DONE] Dataset: {out}/")
    print(f"       annotations/train.json  val.json  test.json")
    print(f"       dataset_info.json")


if __name__ == "__main__":
    main()