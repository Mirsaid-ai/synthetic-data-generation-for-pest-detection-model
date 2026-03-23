"""
run_pipeline.py
===============
All-in-one pipeline:
  depth map → floor mask → configs → render videos + COCO annotations → extract frames → dataset

Usage:
    # Generate 20 random videos + full COCO dataset:
    python run_pipeline.py --image kitchen1.png --n 20 --output_dir out/

    # Use a single hand-crafted config:
    python run_pipeline.py --image kitchen1.png --config my_config.json --output_dir out/

    # Skip steps already done:
    python run_pipeline.py --image kitchen1.png --n 10 --output_dir out/ \
                           --skip_depth --skip_mask

    # Skip frame extraction (videos only, no dataset):
    python run_pipeline.py --image kitchen1.png --n 10 --output_dir out/ \
                           --skip_extract

Full arguments:
    --image           Kitchen image path (required)
    --output_dir      Root output directory (default: pipeline_out/)
    --n               Number of random configs/videos to generate (default: 10)
    --config          Path to a single config JSON — skips random generation
    --mice            Min max mice range (default: 0 3)
    --cockroaches     Min max cockroach range (default: 0 5)
    --duration        Video duration range seconds (default: 15 30)
    --fps             Frames per second (default: 25)
    --jobs            Parallel render jobs (default: 1)
    --floor_labels    ADE20K floor label indices (default: 3)
    --depth_thresh    Depth threshold 0-255 (default: 40)
    --split           Train/val/test fractions (default: 0.8 0.1 0.1)
    --every_n         Extract every Nth frame for dataset (default: 1)
    --no_empty_frames Skip frames with no pest annotations in dataset
    --skip_depth      Skip depth map generation (if already exists)
    --skip_mask       Skip floor mask generation (if already exists)
    --skip_configs    Skip config generation (if already exists in configs/)
    --skip_extract    Skip frame extraction and dataset assembly
    --debug_mask      Save debug overlay for the floor mask
    --seed            Random seed for config generation and dataset split
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))

def _find_extract_script(here):
    """Support both extract_frames.py and extract_frame.py filenames."""
    for name in ("extract_frames.py", "extract_frame.py"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            return path
    return os.path.join(here, "extract_frames.py")   # default (will error clearly)

SCRIPTS = {
    "depth":   os.path.join(HERE, "generate_depth_map.py"),
    "mask":    os.path.join(HERE, "generate_floor_mask.py"),
    "configs": os.path.join(HERE, "generate_configs.py"),
    "render":  os.path.join(HERE, "batch_render.py"),
    "single":  os.path.join(HERE, "add_pests_to_kitchen.py"),
    "extract": _find_extract_script(HERE),
}


def run(cmd, step_name):
    print(f"\n{'═'*60}")
    print(f"  STEP: {step_name}")
    print(f"{'═'*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Full pest video generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image",         default=None,
                        help="Kitchen image path. If --config is provided, read from config.")
    parser.add_argument("--output_dir",    default="pipeline_out")
    parser.add_argument("--n",             type=int, default=10)
    parser.add_argument("--config",        default=None,
                        help="Single config JSON — skips random config generation")
    parser.add_argument("--mice",          type=int, nargs=2, default=[0, 3])
    parser.add_argument("--cockroaches",   type=int, nargs=2, default=[0, 5])
    parser.add_argument("--rats",          type=int, nargs=2, default=[0, 2])
    parser.add_argument("--duration",      type=float, nargs=2, default=[15, 30])
    parser.add_argument("--fps",           type=int, default=25)
    parser.add_argument("--jobs",          type=int, default=1)
    parser.add_argument("--floor_labels",  type=int, nargs="+", default=[3])
    parser.add_argument("--depth_thresh",  type=float, default=40)
    parser.add_argument("--skip_depth",    action="store_true")
    parser.add_argument("--skip_mask",     action="store_true")
    parser.add_argument("--skip_configs",  action="store_true")
    parser.add_argument("--debug_mask",    action="store_true")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--split",         type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--every_n",       type=int, default=1,
                        help="Extract every Nth frame for dataset (default: 1)")
    parser.add_argument("--no_empty_frames", action="store_true",
                        help="Skip frames with no pest annotations in dataset")
    parser.add_argument("--skip_extract",  action="store_true",
                        help="Skip frame extraction and dataset assembly")
    args = parser.parse_args()

    # ── Resolve image ────────────────────────────────────────────────
    # If --image not given but --config is, read image path from the config
    if args.image is None:
        if args.config is None:
            print("[ERROR] Provide --image, or --config containing an 'image' field")
            sys.exit(1)
        try:
            with open(args.config) as f:
                _cfg_peek = json.load(f)
            args.image = _cfg_peek.get("image")
            if not args.image:
                print("[ERROR] Config has no 'image' field and --image was not provided")
                sys.exit(1)
            print(f"[INFO] Image read from config: {args.image}")
        except Exception as e:
            print(f"[ERROR] Could not read config: {e}")
            sys.exit(1)

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    image_stem = Path(args.image).stem
    # Scope everything under out/<image_stem>/ so multiple images never collide
    out        = os.path.join(args.output_dir, image_stem)
    os.makedirs(out, exist_ok=True)

    depth_path   = os.path.join(out, f"{image_stem}_depth.png")
    mask_path    = os.path.join(out, f"{image_stem}_mask.png")
    config_dir   = os.path.join(out, "configs")
    video_dir    = os.path.join(out, "videos")
    dataset_dir  = os.path.join(out, "dataset")
    os.makedirs(video_dir, exist_ok=True)

    print(f"\n[PIPELINE] Image : {args.image}")
    print(f"[PIPELINE] Output: {out}/")

    # ── Step 1: Depth map ────────────────────────────────────────────
    if args.skip_depth and os.path.exists(depth_path):
        print(f"\n[SKIP] Depth map exists: {depth_path}")
    else:
        if not os.path.exists(SCRIPTS["depth"]):
            print(f"[WARN] Depth script not found: {SCRIPTS['depth']}")
            print("       Skipping depth generation. "
                  "Run your own depth script and place output at:")
            print(f"       {depth_path}")
            depth_path = None
        else:
            run([sys.executable, SCRIPTS["depth"],
                 "--image",  args.image,
                 "--output", depth_path],
                "Generate depth map")
            if not Path(depth_path).exists():
                print(f"[WARN] Depth map not found at {depth_path}")
                depth_path = None

    # ── Step 2: Floor mask ───────────────────────────────────────────
    if args.skip_mask and os.path.exists(mask_path):
        print(f"\n[SKIP] Floor mask exists: {mask_path}")
    elif not args.skip_mask or not os.path.exists(mask_path):
        mask_cmd = [
            sys.executable, SCRIPTS["mask"],
            "--image",        args.image,
            "--output",       mask_path,
            "--floor_labels", *[str(l) for l in args.floor_labels],
            "--depth_thresh", str(args.depth_thresh),
        ]
        if depth_path and os.path.exists(depth_path):
            mask_cmd += ["--depth", depth_path]
        if args.debug_mask:
            mask_cmd += ["--debug"]
        run(mask_cmd, "Generate floor mask")

    # ── Step 3: Configs or single config ─────────────────────────────
    if args.config:
        # Single config mode — render just this one
        print(f"\n[INFO] Single config mode: {args.config}")

        # Patch the config — always use pipeline-computed mask/depth paths
        # so relative paths in the original config don't break the render step
        with open(args.config) as f:
            cfg = json.load(f)
        if os.path.exists(mask_path):
            cfg["mask"] = mask_path
        if depth_path and os.path.exists(depth_path):
            cfg["depth"] = depth_path
        if "output" not in cfg:
            cfg["output"] = os.path.join(video_dir, f"{image_stem}_output.mp4")
        else:
            # Prefix with image_stem to avoid collisions across images
            base = os.path.basename(cfg["output"])
            if not base.startswith(image_stem):
                base = f"{image_stem}_{base}"
            cfg["output"] = os.path.join(video_dir, base)

        patched = os.path.join(out, "_single_config.json")
        with open(patched, "w") as f:
            json.dump(cfg, f, indent=2)

        run([sys.executable, SCRIPTS["single"], "--config", patched],
            "Render single video")

        # Extract frames from the single video into the dataset dir
        if args.skip_extract:
            print(f"\n[SKIP] Frame extraction skipped (--skip_extract)")
        else:
            video_out = cfg["output"]
            extract_cmd = [
                sys.executable, SCRIPTS["extract"],
                "--video",      video_out,
                "--output_dir", dataset_dir,
                "--split",      str(args.split[0]), str(args.split[1]), str(args.split[2]),
                "--every_n",    str(args.every_n),
                "--quality",    "95",
            ]
            if args.no_empty_frames:
                extract_cmd += ["--no_empty"]
            if args.seed is not None:
                extract_cmd += ["--seed", str(args.seed)]
            run(extract_cmd, "Extract frames + build COCO dataset")

        print(f"\n{'═'*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Video   → {cfg['output']}")
        if not args.skip_extract:
            print(f"  Dataset → {dataset_dir}/")
            print(f"            annotations/train.json  val.json  test.json")
        print(f"{'═'*60}\n")
        return

    # Random config generation
    if args.skip_configs and os.path.isdir(config_dir) and \
       any(Path(config_dir).glob("*.json")):
        print(f"\n[SKIP] Configs exist in: {config_dir}")
    else:
        os.makedirs(config_dir, exist_ok=True)
        cfg_cmd = [
            sys.executable, SCRIPTS["configs"],
            "--image",       args.image,
            "--output_dir",  config_dir,
            "--n",           str(args.n),
            "--mice",        str(args.mice[0]),        str(args.mice[1]),
            "--cockroaches", str(args.cockroaches[0]), str(args.cockroaches[1]),
            "--rats",        str(args.rats[0]),        str(args.rats[1]),
            "--duration",    str(args.duration[0]),    str(args.duration[1]),
            "--fps",         str(args.fps),
            "--output_prefix", f"{image_stem}_video",
        ]
        if os.path.exists(mask_path):
            cfg_cmd += ["--mask", mask_path]
        if depth_path and os.path.exists(depth_path):
            cfg_cmd += ["--depth", depth_path]
        if args.seed is not None:
            cfg_cmd += ["--seed", str(args.seed)]
        run(cfg_cmd, f"Generate {args.n} random configs")

    # ── Step 4: Batch render ─────────────────────────────────────────
    run([
        sys.executable, SCRIPTS["render"],
        "--config_dir",  config_dir,
        "--output_dir",  video_dir,
        "--jobs",        str(args.jobs),
    ], f"Batch render (jobs={args.jobs})")

    # ── Step 5: Extract frames + build COCO dataset ───────────────────
    if args.skip_extract:
        print(f"\n[SKIP] Frame extraction skipped (--skip_extract)")
    else:
        extract_cmd = [
            sys.executable, SCRIPTS["extract"],
            "--video_dir",  video_dir,
            "--output_dir", dataset_dir,
            "--split",      str(args.split[0]), str(args.split[1]), str(args.split[2]),
            "--every_n",    str(args.every_n),
            "--quality",    "95",
        ]
        if args.no_empty_frames:
            extract_cmd += ["--no_empty"]
        if args.seed is not None:
            extract_cmd += ["--seed", str(args.seed)]
        run(extract_cmd, "Extract frames + build COCO dataset")

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Videos  → {video_dir}/")
    if not args.skip_extract:
        print(f"  Dataset → {dataset_dir}/")
        print(f"            annotations/train.json  val.json  test.json")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()