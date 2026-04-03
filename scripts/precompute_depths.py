"""Pre-compute depth maps and surface normals for all kitchen images.

Run this LOCALLY on M3 Pro (MPS-accelerated) before uploading to Google Drive.
Saves one .npz file per image that Colab can load instead of running Metric3D.

This is the single biggest Colab speedup — Metric3D on CPU takes 5-20 minutes
per image. Pre-computing for 150 images on M3 Pro takes ~30-60 minutes total,
then each Colab video render drops from 5-20 min → ~30 sec for the depth step.

Usage:
    python scripts/precompute_depths.py
    python scripts/precompute_depths.py --image_dir generator/kitchen_img/curated_img
    python scripts/precompute_depths.py --image_dir generator/kitchen_img/curated_img \\
                                        --cache_dir depth_cache/ --workers 1

Output:
    depth_cache/
    ├── kitchen_0002.npz      # depth, normals, fx, gravity per image
    ├── kitchen_0003.npz
    └── ...
    └── manifest.json         # maps image filename → cache filename

Upload the entire depth_cache/ folder to Google Drive alongside the code.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add mmcv_stub and repo root to path before importing generator modules
_REPO_ROOT = Path(__file__).parent.parent
_MMCV_STUB = _REPO_ROOT / "generator" / "mmcv_stub"
for _p in (str(_MMCV_STUB), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_for_image(image_path: Path) -> dict:
    """Run Metric3D + gravity estimation for one kitchen image."""
    # Import here so the script can do --dry_run / --list without loading models
    from generator.depth_estimator import (
        estimate_gravity,
        estimate_metric3d,
        compute_inference_strategy,
    )
    from concurrent.futures import ThreadPoolExecutor

    strategy = compute_inference_strategy()
    if strategy == "parallel":
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_m3d  = ex.submit(estimate_metric3d, str(image_path))
            f_grav = ex.submit(estimate_gravity,  str(image_path))
        m3d_result  = f_m3d.result()
        grav_result = f_grav.result()
    else:
        m3d_result  = estimate_metric3d(str(image_path))
        grav_result = estimate_gravity(str(image_path))

    return {
        "depth":              m3d_result["depth"].astype(np.float32),
        "normals":            m3d_result["normals"].astype(np.float32),
        "fx":                 float(m3d_result["fx"]),
        # Gravity dict — store the numpy array as float32 for portability
        "gravity_cam":        np.array(grav_result["gravity_cam"], dtype=np.float32),
        "gravity_confidence": float(grav_result["confidence"]),
    }


def load_cached(cache_path: Path) -> dict:
    """Load a .npz cache file and reconstruct the gravity dict expected by pipeline.py."""
    data = np.load(str(cache_path), allow_pickle=False)
    return {
        "depth":   data["depth"],
        "normals": data["normals"],
        "fx":      float(data["fx"]),
        "gravity": {
            "gravity_cam": data["gravity_cam"],
            "confidence":  float(data["gravity_confidence"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Metric3D depth/normals for kitchen images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        default="generator/kitchen_img/curated_img",
        help="Directory of kitchen images (default: generator/kitchen_img/curated_img)",
    )
    parser.add_argument(
        "--cache_dir",
        default="depth_cache",
        help="Output directory for .npz cache files (default: depth_cache/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if a cache file already exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List images that would be processed without running anything",
    )
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    image_dir  = (repo_root / args.image_dir).resolve()
    cache_dir  = (repo_root / args.cache_dir).resolve()

    if not image_dir.exists():
        print(f"Error: image directory not found: {image_dir}")
        sys.exit(1)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which images need processing
    todo = []
    skip = 0
    for img in images:
        cache_path = cache_dir / (img.stem + ".npz")
        if cache_path.exists() and not args.overwrite:
            skip += 1
        else:
            todo.append(img)

    print(f"Found {len(images)} image(s) in {image_dir}")
    print(f"  Already cached: {skip}  |  To process: {len(todo)}")
    if skip and not args.overwrite:
        print("  (use --overwrite to recompute cached files)\n")

    if args.dry_run:
        for p in todo:
            print(f"  [dry-run] would process: {p.name}")
        return

    if not todo:
        print("Nothing to do.")
    else:
        print(f"\nProcessing {len(todo)} image(s) on "
              f"{'MPS' if _has_mps() else 'CPU'}...\n")

    manifest = {}
    ok, failed = 0, []

    # Load existing manifest if present
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    for i, img_path in enumerate(todo, 1):
        cache_path = cache_dir / (img_path.stem + ".npz")
        print(f"[{i:3d}/{len(todo)}] {img_path.name} ...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            result = compute_for_image(img_path)
            np.savez_compressed(
                str(cache_path),
                depth=result["depth"],
                normals=result["normals"],
                fx=np.array(result["fx"], dtype=np.float32),
                gravity_cam=result["gravity_cam"],
                gravity_confidence=np.array(result["gravity_confidence"], dtype=np.float32),
            )
            elapsed = time.monotonic() - t0
            h, w = result["depth"].shape
            print(f"done  ({elapsed:.1f}s, depth {w}×{h}, fx={result['fx']:.1f}px)")
            manifest[img_path.name] = cache_path.name
            ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(img_path.name)

    # Also add already-cached files to manifest
    for img in images:
        if img.name not in manifest:
            cache_path = cache_dir / (img.stem + ".npz")
            if cache_path.exists():
                manifest[img.name] = cache_path.name

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. {ok} processed, {skip} already cached, {len(failed)} failed.")
    print(f"Cache directory: {cache_dir}")
    print(f"Manifest: {manifest_path}")
    if failed:
        print(f"Failed: {failed}")
    print(
        f"\nUpload '{cache_dir.name}/' to Google Drive alongside your code "
        "and pass --depth_cache_dir to render_batch_colab.py."
    )


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
