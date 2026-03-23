"""
generate_configs.py
===================
Randomly generates a set of pest video config files for a given kitchen image.

Usage:
    python generate_configs.py --image kitchen1.png \
                               --mask kitchen1_mask.png \
                               --depth kitchen1_depth.png \
                               --output_dir configs/ \
                               --n 20 \
                               --mice 0 3 \
                               --cockroaches 0 5

Arguments:
    --image         Kitchen image path
    --mask          Floor mask PNG (from generate_floor_mask.py)
    --depth         Depth map PNG (optional)
    --output_dir    Directory to write config JSON files (created if needed)
    --n             Number of configs to generate (default: 10)
    --mice          Min and max number of mice per video  (default: 0 3)
    --cockroaches   Min and max cockroaches per video     (default: 0 5)
    --duration      Video duration range in seconds       (default: 15 30)
    --fps           Frames per second                     (default: 25)
    --output_prefix Prefix for output video filenames     (default: output)
    --seed          Random seed for reproducibility       (optional)
"""

import argparse
import json
import os
import random


# Reasonable size/speed ranges per pest type
PEST_DEFAULTS = {
    "mouse": {
        "size_range":  (40, 70),
        "speed_range": (4, 9),
    },
    "cockroach": {
        "size_range":  (20, 45),
        "speed_range": (6, 13),
    },
}


def random_pest_entry(ptype, count):
    d = PEST_DEFAULTS[ptype]
    return {
        "type":  ptype,
        "count": count,
        "size":  random.randint(*d["size_range"]),
        "speed": round(random.uniform(*d["speed_range"]), 1),
    }


def generate_config(image, mask, depth, output_video, duration, fps):
    """Build one config dict with a random mix of pests."""
    # Randomly decide how many of each pest (caller already sampled counts)
    return None  # built by caller


def main():
    parser = argparse.ArgumentParser(
        description="Generate random pest video configs for a kitchen image.")
    parser.add_argument("--image",         required=True)
    parser.add_argument("--mask",          default=None)
    parser.add_argument("--depth",         default=None)
    parser.add_argument("--output_dir",    default="configs")
    parser.add_argument("--n",             type=int, default=10,
                        help="Number of configs to generate (default: 10)")
    parser.add_argument("--mice",          type=int, nargs=2, default=[0, 3],
                        metavar=("MIN", "MAX"),
                        help="Min/max mice per video (default: 0 3)")
    parser.add_argument("--cockroaches",   type=int, nargs=2, default=[0, 5],
                        metavar=("MIN", "MAX"),
                        help="Min/max cockroaches per video (default: 0 5)")
    parser.add_argument("--duration",      type=float, nargs=2, default=[15, 30],
                        metavar=("MIN", "MAX"),
                        help="Duration range in seconds (default: 15 30)")
    parser.add_argument("--fps",           type=int, default=25)
    parser.add_argument("--output_prefix", default="output",
                        help="Prefix for generated video filenames (default: output)")
    parser.add_argument("--seed",          type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate counts
    if args.mice[0] > args.mice[1]:
        print("[ERROR] --mice MIN must be <= MAX"); raise SystemExit(1)
    if args.cockroaches[0] > args.cockroaches[1]:
        print("[ERROR] --cockroaches MIN must be <= MAX"); raise SystemExit(1)

    generated = 0
    skipped   = 0

    for i in range(args.n):
        n_mice        = random.randint(args.mice[0],        args.mice[1])
        n_cockroaches = random.randint(args.cockroaches[0], args.cockroaches[1])

        # Skip configs with no pests at all
        if n_mice == 0 and n_cockroaches == 0:
            skipped += 1
            # Force at least one pest
            if args.mice[1] > 0:
                n_mice = random.randint(1, args.mice[1])
            else:
                n_cockroaches = random.randint(1, max(1, args.cockroaches[1]))

        duration = round(random.uniform(args.duration[0], args.duration[1]), 1)

        pests = []
        if n_mice > 0:
            pests.append(random_pest_entry("mouse", n_mice))
        if n_cockroaches > 0:
            pests.append(random_pest_entry("cockroach", n_cockroaches))

        video_name = f"{args.output_prefix}_{i:04d}.mp4"

        cfg = {
            "image":    args.image,
            "output":   video_name,
            "duration": duration,
            "fps":      args.fps,
            "pests":    pests,
        }
        if args.mask:
            cfg["mask"]  = args.mask
        if args.depth:
            cfg["depth"] = args.depth

        cfg_path = os.path.join(args.output_dir, f"config_{i:04d}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        total_pests = n_mice + n_cockroaches
        print(f"[{i+1:4d}/{args.n}] {cfg_path}  "
              f"mice={n_mice}  roaches={n_cockroaches}  "
              f"duration={duration}s  → {video_name}")
        generated += 1

    print(f"\n[DONE] {generated} configs written to: {args.output_dir}/")
    if skipped:
        print(f"       ({skipped} zero-pest draws were corrected to have ≥1 pest)")


if __name__ == "__main__":
    main()