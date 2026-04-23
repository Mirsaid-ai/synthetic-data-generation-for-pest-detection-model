"""Batch video renderer for Google Colab / Google Drive.

Renders N synthetic pest videos per kitchen image, using pre-computed depth
caches to skip Metric3D inference on Colab CPU.

Kitchen-level train/val/test split is applied HERE, before rendering, so
no background image ever appears in more than one split. This prevents the
data leakage that would occur if you split at the frame level.

Split design:
    10% of kitchens → HELD-OUT (never rendered — reserved for final eval)
    72% of kitchens → TRAIN videos
     9% of kitchens → VAL videos
    (10% held-out rendered separately via --held_out_mode for test set)

Usage on Google Colab:
    # 1. Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # 2. Install deps
    !pip install -q ...

    # 3. Run (adjust paths to your Drive layout)
    !python scripts/render_batch_colab.py \\
        --image_dir   /content/drive/MyDrive/pest_project/kitchens \\
        --depth_cache /content/drive/MyDrive/pest_project/depth_cache \\
        --output_dir  /content/drive/MyDrive/pest_project/renders \\
        --n           20 \\
        --session_id  0 \\
        --total_sessions 5

    # 4. Each session handles a slice of kitchens (for parallel Colab runs).

Usage locally for testing:
    python scripts/render_batch_colab.py --image_dir generator/kitchen_img/curated_img \\
        --n 2 --dry_run

Output structure:
    renders/
    ├── manifest.json          ← maps kitchen → split + list of job_ids
    ├── train/
    │   ├── <job_id>/
    │   │   ├── frames/frame_0001.png ...
    │   │   └── labels/annotations.json
    │   └── ...
    ├── val/
    │   └── ...
    └── held_out/              ← only populated with --held_out_mode
        └── ...
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Metric3D requires mmcv which is not pip-installable on all platforms.
# The repo ships a minimal stub at generator/mmcv_stub/ — add it to sys.path
# before any generator import so the stub is found first.
_REPO_ROOT = Path(__file__).parent.parent
_MMCV_STUB = _REPO_ROOT / "generator" / "mmcv_stub"
if str(_MMCV_STUB) not in sys.path:
    sys.path.insert(0, str(_MMCV_STUB))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "held_out")

# Fraction of kitchens reserved for each role (must sum to 1.0)
HELD_OUT_FRAC = 0.10   # never rendered in main run — used for final test eval
VAL_FRAC      = 0.10   # rendered, used for validation during training
# Train gets the rest: 1 - HELD_OUT_FRAC - VAL_FRAC = 0.80


def assign_kitchen_splits(kitchens: list[Path], seed: int) -> dict[str, list[Path]]:
    """Deterministically split a kitchen list into train / val / held_out.

    Args:
        kitchens: Sorted list of kitchen image paths.
        seed:     Random seed — must be the same across all Colab sessions.

    Returns:
        Dict mapping split name → list of kitchen paths in that split.
    """
    rng = random.Random(seed)
    shuffled = list(kitchens)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_held = max(1, round(n * HELD_OUT_FRAC))
    n_val  = max(1, round(n * VAL_FRAC))
    n_train = n - n_held - n_val

    return {
        "train":    shuffled[:n_train],
        "val":      shuffled[n_train:n_train + n_val],
        "held_out": shuffled[n_train + n_val:],
    }


def load_depth_cache(cache_dir: Path, image_path: Path) -> dict | None:
    """Load pre-computed depth/normals for an image. Returns None if missing."""
    if cache_dir is None:
        return None
    npz_path = cache_dir / (image_path.stem + ".npz")
    if not npz_path.exists():
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=False)
        return {
            "depth":   data["depth"].astype(np.float32),
            "normals": data["normals"].astype(np.float32),
            "fx":      float(data["fx"]),
            "gravity": {
                "gravity_cam": data["gravity_cam"],
                "confidence":  float(data["gravity_confidence"]),
            },
        }
    except Exception as e:
        print(f"  [WARN] Could not load depth cache for {image_path.name}: {e}")
        return None


def render_kitchen(
    image_path: Path,
    split: str,
    output_dir: Path,
    n_videos: int,
    depth_cache: dict | None,
    dry_run: bool = False,
) -> list[str]:
    """Render N videos from one kitchen image. Returns list of job_ids."""
    if dry_run:
        fake_ids = [f"dry_{image_path.stem}_{i:03d}" for i in range(n_videos)]
        print(f"  [dry-run] would render {n_videos} videos → {split}/")
        return fake_ids

    # Import pipeline lazily so --dry_run / --list work without loading models
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generator.pipeline import generate_video

    split_dir = output_dir / split
    frames_root = split_dir / "frames"
    labels_root = split_dir / "labels"
    videos_root = split_dir / "videos"
    for d in (frames_root, labels_root, videos_root):
        d.mkdir(parents=True, exist_ok=True)

    job_ids = []
    for i in range(n_videos):
        t0 = time.monotonic()
        try:
            result = generate_video(
                image_path=str(image_path),
                frames_root=str(frames_root),
                labels_root=str(labels_root),
                videos_root=str(videos_root),
                assemble_video=True,
                frame_format="jpg",
                save_scene_previews=False,
                save_mask_previews=False,
                save_movement_masks=False,
                keep_only_frame_outputs=True,
                save_every_n=2,             # save every 2nd frame → 50% storage saving
                precomputed_depth=depth_cache,
            )
            job_ids.append(result["job_id"])
            elapsed = time.monotonic() - t0
            pests = result["pest_counts"]
            pest_str = ", ".join(f"{k}={v}" for k, v in pests.items() if v > 0) or "none"
            print(f"    [{i+1}/{n_videos}] job={result['job_id']}  "
                  f"pests={pest_str}  frames={result['saved_num_frames']}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"    [{i+1}/{n_videos}] ERROR: {e}")

    return job_ids


def main():
    parser = argparse.ArgumentParser(
        description="Batch render synthetic pest videos for Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image_dir", required=True,
                        help="Directory of kitchen images")
    parser.add_argument("--output_dir", default="renders",
                        help="Root output directory (default: renders/)")
    parser.add_argument("--depth_cache", default=None,
                        help="Directory of pre-computed .npz depth caches "
                             "(from precompute_depths.py). Strongly recommended "
                             "for CPU-only Colab sessions.")
    parser.add_argument("--n", type=int, default=20,
                        help="Videos per kitchen (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for kitchen split (must match across "
                             "all Colab sessions, default: 42)")
    parser.add_argument("--session_id", type=int, default=0,
                        help="This session's index (0-based) when running "
                             "multiple Colab sessions in parallel (default: 0)")
    parser.add_argument("--total_sessions", type=int, default=1,
                        help="Total number of parallel Colab sessions (default: 1)")
    parser.add_argument("--held_out_mode", action="store_true",
                        help="Render only the held-out kitchens (for test set "
                             "generation after training). Do NOT use during main "
                             "batch rendering.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be rendered without doing anything")
    parser.add_argument("--list_splits", action="store_true",
                        help="Print the kitchen split and exit")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    cache_dir  = Path(args.depth_cache) if args.depth_cache else None

    if not image_dir.exists():
        print(f"Error: image_dir not found: {image_dir}"); sys.exit(1)
    if cache_dir and not cache_dir.exists():
        print(f"Warning: depth_cache dir not found: {cache_dir}. "
              "Will run Metric3D for every image (slow on CPU).")
        cache_dir = None

    kitchens = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not kitchens:
        print(f"No images found in {image_dir}"); sys.exit(1)

    splits = assign_kitchen_splits(kitchens, seed=args.seed)

    if args.list_splits:
        for split_name, ks in splits.items():
            print(f"\n{split_name} ({len(ks)} kitchens):")
            for k in ks:
                print(f"  {k.name}")
        return

    # Choose which kitchens this session renders
    if args.held_out_mode:
        active_split   = "held_out"
        active_kitchens = splits["held_out"]
    else:
        active_kitchens = splits["train"] + splits["val"]

    # Slice for this Colab session
    session_kitchens = [
        k for i, k in enumerate(active_kitchens)
        if i % args.total_sessions == args.session_id
    ]

    print(f"Kitchen split summary:")
    print(f"  Total kitchens:   {len(kitchens)}")
    print(f"  Train:            {len(splits['train'])} kitchens × {args.n} = "
          f"{len(splits['train']) * args.n} videos")
    print(f"  Val:              {len(splits['val'])} kitchens × {args.n} = "
          f"{len(splits['val']) * args.n} videos")
    print(f"  Held-out (test):  {len(splits['held_out'])} kitchens "
          f"(not rendered in main run)")
    print(f"\nThis session ({args.session_id+1}/{args.total_sessions}): "
          f"{len(session_kitchens)} kitchens to render")
    print(f"Depth cache: {'enabled' if cache_dir else 'DISABLED (slow!)'}")
    print(f"Output: {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialise manifest
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "seed": args.seed,
            "n_per_kitchen": args.n,
            "split_fracs": {
                "held_out": HELD_OUT_FRAC,
                "val": VAL_FRAC,
                "train": round(1.0 - HELD_OUT_FRAC - VAL_FRAC, 4),
            },
            "kitchens": {},
        }

    # Render each kitchen in this session's slice
    total_ok = 0
    total_skipped = 0
    for ki, kitchen_path in enumerate(session_kitchens, 1):
        split = "train" if kitchen_path in splits["train"] else (
                "val"   if kitchen_path in splits["val"]   else "held_out")

        key = kitchen_path.name

        # Skip kitchens that are already fully rendered in the manifest.
        # This makes re-runs safe even if TOTAL_SESSIONS changes between runs.
        already_done = len(manifest["kitchens"].get(key, {}).get("job_ids", []))
        if already_done >= args.n:
            print(f"[{ki}/{len(session_kitchens)}] {kitchen_path.name}  ({split})  "
                  f"[skip — {already_done} videos already in manifest]")
            total_skipped += 1
            continue

        print(f"[{ki}/{len(session_kitchens)}] {kitchen_path.name}  ({split})")

        depth_cache = load_depth_cache(cache_dir, kitchen_path) if cache_dir else None
        if cache_dir and depth_cache is None:
            print(f"  [WARN] No depth cache — running Metric3D (slow on CPU)")

        job_ids = render_kitchen(
            image_path=kitchen_path,
            split=split,
            output_dir=output_dir,
            n_videos=args.n,
            depth_cache=depth_cache,
            dry_run=args.dry_run,
        )
        total_ok += len(job_ids)

        # Update manifest entry for this kitchen
        if key not in manifest["kitchens"]:
            manifest["kitchens"][key] = {"split": split, "job_ids": []}
        manifest["kitchens"][key]["job_ids"].extend(job_ids)

        # Persist manifest after every kitchen so partial runs are resumable
        if not args.dry_run:
            manifest_path.write_text(json.dumps(manifest, indent=2))

    if total_skipped:
        print(f"\n{total_skipped} kitchens skipped (already fully rendered).")

    print(f"\nSession done.  {total_ok} videos rendered.")
    print(f"Manifest: {manifest_path}")
    print("\nNext step:")
    print("  python scripts/build_dataset.py --render_dir renders/ --output_dir pest_dataset/")


if __name__ == "__main__":
    main()
