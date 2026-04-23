"""Build the final YOLOv8 training dataset from rendered videos.

Reads the manifest written by render_batch_colab.py, which already encodes the
kitchen-level train/val split. Frames from the same kitchen image are NEVER
split across train and val — this prevents data leakage.

What this script does:
  1. Reads manifest.json to get the kitchen → split mapping
  2. Collects all rendered frames + COCO annotations per split
  3. Converts COCO bbox annotations → YOLOv8 .txt label files
  4. Copies images to pest_dataset/images/{train,val,test}/
  5. Writes pest_dataset/labels/{train,val,test}/*.txt
  6. Writes pest_dataset/data.yaml (ready for: yolo train data=data.yaml)
  7. Prints dataset statistics

Usage:
    # After render_batch_colab.py completes:
    python scripts/build_dataset.py --render_dir renders/ --output_dir pest_dataset/

    # Extract every 2nd frame (halves dataset size, still ample data):
    python scripts/build_dataset.py --render_dir renders/ --output_dir pest_dataset/ \\
        --every_n 2

    # Also include held-out renders as the test split:
    python scripts/build_dataset.py --render_dir renders/ --output_dir pest_dataset/ \\
        --test_render_dir renders_held_out/

Output:
    pest_dataset/
    ├── images/
    │   ├── train/   frame images (.jpg)
    │   ├── val/
    │   └── test/    (only if --test_render_dir provided)
    ├── labels/
    │   ├── train/   YOLOv8 .txt files (one per frame)
    │   ├── val/
    │   └── test/
    └── data.yaml    → pass directly to: yolo train data=data.yaml ...
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ── Class definitions (must match generator/compositing.py _CATEGORY_MAP) ─────
CATEGORIES = [
    {"id": 1, "name": "mouse",     "yolo_id": 0},
    {"id": 2, "name": "rat",       "yolo_id": 1},
    {"id": 3, "name": "cockroach", "yolo_id": 2},
]
COCO_TO_YOLO_ID = {c["id"]: c["yolo_id"] for c in CATEGORIES}
CLASS_NAMES     = [c["name"] for c in sorted(CATEGORIES, key=lambda c: c["yolo_id"])]


# ── COCO → YOLO conversion ─────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert COCO [x, y, w, h] (absolute pixels) → YOLO normalised [cx, cy, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def write_yolo_label(ann_path: Path, annotations: list, img_w: int, img_h: int) -> int:
    """Write a YOLOv8 .txt label file. Returns number of annotations written."""
    lines = []
    for ann in annotations:
        yolo_id = COCO_TO_YOLO_ID.get(ann["category_id"])
        if yolo_id is None:
            continue
        if ann.get("iscrowd", 0):
            continue
        cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        if nw < 1e-4 or nh < 1e-4:
            continue
        lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    ann_path.write_text("\n".join(lines))
    return len(lines)


# ── Frame + annotation collection ─────────────────────────────────────────────

def collect_job_frames(
    job_frames_dir: Path,
    job_labels_dir: Path,
    every_n: int = 1,
) -> list[dict]:
    """Collect frame records from one rendered job directory.

    Returns list of dicts: {frame_path, img_w, img_h, annotations: [...]}
    """
    ann_path = job_labels_dir / "annotations.json"
    if not ann_path.exists():
        return []

    try:
        coco = json.loads(ann_path.read_text())
    except Exception:
        return []

    # Index annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_img[ann["image_id"]].append(ann)

    records = []
    for idx, img_rec in enumerate(coco.get("images", [])):
        if every_n > 1 and idx % every_n != 0:
            continue
        fname  = img_rec["file_name"]
        fpath  = job_frames_dir / fname
        if not fpath.exists():
            continue
        records.append({
            "frame_path":  fpath,
            "img_w":       img_rec.get("width",  640),
            "img_h":       img_rec.get("height", 480),
            "annotations": ann_by_img.get(img_rec["id"], []),
        })

    return records


# ── data.yaml writer ───────────────────────────────────────────────────────────

def write_data_yaml(output_dir: Path, splits: list[str]) -> Path:
    """Write YOLOv8-compatible data.yaml."""
    yaml_lines = [
        f"path: {output_dir.resolve()}",
        f"train: images/train",
        f"val:   images/val",
    ]
    if "test" in splits:
        yaml_lines.append("test:  images/test")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(CLASS_NAMES)}")
    yaml_lines.append(f"names: {CLASS_NAMES}")
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return yaml_path


# ── Per-split processing ───────────────────────────────────────────────────────

def process_split(
    split: str,
    job_dirs: list[tuple[Path, Path]],   # [(frames_dir, labels_dir), ...]
    output_dir: Path,
    every_n: int,
    stats: dict,
    resume: bool = False,
    use_symlinks: bool = False,
):
    """Copy (or symlink) frames and write YOLO labels for one split."""
    img_out_dir   = output_dir / "images" / split
    label_out_dir = output_dir / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_skipped = 0
    n_annotations = 0
    per_class = defaultdict(int)

    total_jobs = len(job_dirs)
    t0 = time.time()

    iterator = tqdm(job_dirs, desc=split, unit="job") if _TQDM else job_dirs

    for job_idx, (frames_dir, labels_dir) in enumerate(iterator):
        records = collect_job_frames(frames_dir, labels_dir, every_n=every_n)
        job_id  = frames_dir.name

        for rec in records:
            stem = f"{job_id}_{rec['frame_path'].stem}"
            img_dest   = img_out_dir   / f"{stem}.jpg"
            label_dest = label_out_dir / f"{stem}.txt"

            if resume and img_dest.exists() and label_dest.exists():
                n_skipped += 1
                n_images  += 1
                continue

            if use_symlinks:
                src = rec["frame_path"].resolve()
                if img_dest.is_symlink() or img_dest.exists():
                    img_dest.unlink()
                os.symlink(src, img_dest)
            elif rec["frame_path"].suffix.lower() == ".jpg":
                shutil.copy2(rec["frame_path"], img_dest)
            else:
                try:
                    from PIL import Image
                    Image.open(rec["frame_path"]).convert("RGB").save(
                        img_dest, format="JPEG", quality=95)
                except Exception:
                    shutil.copy2(rec["frame_path"], img_dest)

            # Write YOLO label
            n_ann = write_yolo_label(
                label_dest, rec["annotations"], rec["img_w"], rec["img_h"])
            n_images      += 1
            n_annotations += n_ann
            for ann in rec["annotations"]:
                yolo_id = COCO_TO_YOLO_ID.get(ann["category_id"])
                if yolo_id is not None:
                    per_class[CLASS_NAMES[yolo_id]] += 1

        if not _TQDM and (job_idx + 1) % 100 == 0:
            elapsed  = time.time() - t0
            pct      = (job_idx + 1) / total_jobs * 100
            eta_s    = elapsed / (job_idx + 1) * (total_jobs - job_idx - 1)
            eta_min  = eta_s / 60
            print(f"  [{split}] {job_idx+1}/{total_jobs} jobs ({pct:.0f}%)  "
                  f"frames={n_images:,}  skipped={n_skipped:,}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta_min:.1f}m",
                  flush=True)

    elapsed = time.time() - t0
    mode_str = "symlinked" if use_symlinks else "copied"
    skip_note = f"  ({n_skipped:,} already existed, skipped)" if resume and n_skipped else ""
    print(f"  [{split}] done in {elapsed/60:.1f}m — {n_images:,} frames {mode_str}{skip_note}")

    stats[split] = {
        "images":      n_images,
        "annotations": n_annotations,
        "per_class":   dict(per_class),
        "negative_frames": n_images - sum(1 for _ in [0] if n_annotations > 0),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build YOLOv8 dataset from rendered videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--render_dir", required=True,
                        help="Output dir from render_batch_colab.py (contains manifest.json)")
    parser.add_argument("--output_dir", default="pest_dataset",
                        help="Where to write the final YOLO dataset (default: pest_dataset/)")
    parser.add_argument("--test_render_dir", default=None,
                        help="Optional: separate render dir from held-out kitchens "
                             "(generates the test split)")
    parser.add_argument("--every_n", type=int, default=1,
                        help="Use every Nth frame per video (default: 1 = all frames). "
                             "Use 2 to halve dataset size with minimal quality loss.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count frames that would be processed without copying")
    parser.add_argument("--resume", action="store_true",
                        help="Skip frames whose image+label files already exist in output_dir "
                             "(safe to use after an interrupted run)")
    parser.add_argument("--symlink", action="store_true",
                        help="Create symlinks to frame images instead of copying them. "
                             "~100x faster on Google Drive FUSE — avoids reading/writing "
                             "image data entirely. YOLO reads through symlinks at train time.")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    output_dir = Path(args.output_dir)

    manifest_path = render_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {render_dir}")
        print("Run render_batch_colab.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest: seed={manifest.get('seed')}  "
          f"n_per_kitchen={manifest.get('n_per_kitchen')}  "
          f"kitchens={len(manifest.get('kitchens', {}))}")

    # Group job directories by split
    split_jobs: dict[str, list[tuple[Path, Path]]] = defaultdict(list)

    for kitchen_name, info in manifest.get("kitchens", {}).items():
        split   = info["split"]
        if split == "held_out":
            continue  # held-out kitchens are only in test set
        for job_id in info.get("job_ids", []):
            frames_dir = render_dir / split / "frames" / job_id
            labels_dir = render_dir / split / "labels" / job_id
            if frames_dir.exists() and labels_dir.exists():
                split_jobs[split].append((frames_dir, labels_dir))
            else:
                print(f"  [WARN] Missing dirs for job {job_id} — skipping")

    # Optional held-out test split from a separate render run
    if args.test_render_dir:
        test_render_dir = Path(args.test_render_dir)
        test_manifest_path = test_render_dir / "manifest.json"
        if test_manifest_path.exists():
            test_manifest = json.loads(test_manifest_path.read_text())
            for kitchen_name, info in test_manifest.get("kitchens", {}).items():
                for job_id in info.get("job_ids", []):
                    frames_dir = test_render_dir / info["split"] / "frames" / job_id
                    labels_dir = test_render_dir / info["split"] / "labels" / job_id
                    if frames_dir.exists() and labels_dir.exists():
                        split_jobs["test"].append((frames_dir, labels_dir))
        else:
            print(f"[WARN] No manifest.json in test_render_dir — skipping test split")

    active_splits = [s for s in ("train", "val", "test") if split_jobs.get(s)]
    print(f"\nActive splits: {active_splits}")
    for s in active_splits:
        n_jobs = len(split_jobs[s])
        print(f"  {s}: {n_jobs} jobs")

    if args.dry_run:
        print("\n[dry-run] No files will be written.")
        for s in active_splits:
            total = sum(
                len(collect_job_frames(fd, ld, every_n=args.every_n))
                for fd, ld in split_jobs[s]
            )
            print(f"  {s}: ~{total} frames")
        return

    # Warn if output_dir is on a network mount (Google Drive / FUSE)
    # Writing 100K+ small files to Drive triggers quota errors and is ~10x slower.
    out_str = str(output_dir.resolve())
    if any(p in out_str for p in ("/gdrive", "/content/drive", "/mnt/drive")):
        print("\n⚠️  WARNING: output_dir is on Google Drive.")
        print("   Writing 100K+ small files to Drive will hit quota limits and be very slow.")
        print("   Recommended: use a local path like /content/pest_dataset, then tar+copy to Drive.")
        print("   Continuing anyway — use Ctrl+C to abort and re-run with a local output path.\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    if args.resume:
        print("\n[resume mode] Already-copied frames will be skipped.")

    if args.symlink:
        print("\n[symlink mode] Images will be symlinked, not copied (fast on Drive FUSE).")

    for split in active_splits:
        print(f"\nProcessing {split}...")
        process_split(split, split_jobs[split], output_dir, args.every_n, stats,
                      resume=args.resume, use_symlinks=args.symlink)
        s = stats[split]
        print(f"  {s['images']} frames  |  {s['annotations']} annotations  "
              f"|  per-class: {s['per_class']}")

    # Write data.yaml
    yaml_path = write_data_yaml(output_dir, active_splits)
    print(f"\ndata.yaml written: {yaml_path}")

    # Dataset summary
    total_images = sum(s["images"] for s in stats.values())
    total_anns   = sum(s["annotations"] for s in stats.values())
    print(f"\n{'='*55}")
    print(f"Dataset: {output_dir}")
    print(f"  Total frames:      {total_images:,}")
    print(f"  Total annotations: {total_anns:,}")
    for split in active_splits:
        s = stats[split]
        pct = s["images"] / max(total_images, 1) * 100
        print(f"  {split:6s}: {s['images']:7,} frames ({pct:.0f}%)  "
              f"  anns={s['annotations']:,}  per-class={s['per_class']}")
    neg = sum(s["images"] for s in stats.values()) - total_anns
    print(f"\n  Negative frames (no pest): ~{max(0, total_images - total_anns):,}")
    print(f"\nTraining command:")
    print(f"  yolo train \\")
    print(f"    data={yaml_path.resolve()} \\")
    print(f"    model=yolov8m.pt \\")
    print(f"    epochs=100 imgsz=640 batch=16 device=0 \\")
    print(f"    augment=True mosaic=1.0 degrees=10 fliplr=0.5 \\")
    print(f"    hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 translate=0.1 scale=0.3")


if __name__ == "__main__":
    main()
