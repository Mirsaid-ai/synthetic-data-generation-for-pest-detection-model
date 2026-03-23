"""
batch_render.py
===============
Renders all config JSON files in a directory using add_pests_to_kitchen.py.

Usage:
    python batch_render.py --config_dir configs/ --output_dir videos/

Arguments:
    --config_dir    Directory containing config_XXXX.json files
    --output_dir    Directory to write rendered MP4s (default: same as config_dir)
    --jobs          Number of parallel render jobs (default: 1)
                    Increase if you have multiple CPU cores free.
    --skip_existing Skip configs whose output video already exists (default: True)
    --fail_fast     Stop on first render error (default: False)
"""

import argparse
import os
import json
import sys
import subprocess
import concurrent.futures
from pathlib import Path


def find_configs(config_dir):
    configs = sorted(Path(config_dir).glob("*.json"))
    if not configs:
        print(f"[ERROR] No JSON files found in: {config_dir}")
        sys.exit(1)
    return configs


def render_config(cfg_path, output_dir, skip_existing):
    """Render a single config. Returns (cfg_path, success, message)."""
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Redirect output video into output_dir if specified
    video_name = os.path.basename(cfg.get("output", "output.mp4"))
    if output_dir:
        video_path = os.path.join(output_dir, video_name)
        cfg["output"] = video_path
    else:
        video_path = cfg.get("output", "output.mp4")

    if skip_existing and os.path.exists(video_path):
        return cfg_path, True, f"SKIP (exists): {video_path}"

    # Write a temporary config with the updated output path
    tmp_cfg = str(cfg_path) + ".tmp.json"
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "add_pests_to_kitchen.py")

    try:
        result = subprocess.run(
            [sys.executable, script, "--config", tmp_cfg],
            capture_output=True, text=True, timeout=600
        )
        os.remove(tmp_cfg)
        if result.returncode == 0:
            return cfg_path, True, f"OK → {video_path}"
        else:
            err = result.stderr.strip().split("\n")[-1]
            return cfg_path, False, f"FAIL: {err}"
    except subprocess.TimeoutExpired:
        os.remove(tmp_cfg)
        return cfg_path, False, "FAIL: timeout (>10 min)"
    except Exception as e:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
        return cfg_path, False, f"FAIL: {e}"


def main():
    parser = argparse.ArgumentParser(description="Batch render pest videos from configs.")
    parser.add_argument("--config_dir",    required=True)
    parser.add_argument("--output_dir",    default=None,
                        help="Where to save videos (default: alongside configs)")
    parser.add_argument("--jobs",          type=int, default=1,
                        help="Parallel render jobs (default: 1)")
    parser.add_argument("--no_skip",       action="store_true",
                        help="Re-render even if output video already exists")
    parser.add_argument("--fail_fast",     action="store_true",
                        help="Stop on first error")
    args = parser.parse_args()

    configs = find_configs(args.config_dir)
    print(f"[INFO] Found {len(configs)} config(s) in: {args.config_dir}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] Videos → {args.output_dir}")

    skip_existing = not args.no_skip
    succeeded, failed = 0, 0
    errors = []

    if args.jobs == 1:
        for idx, cfg_path in enumerate(configs):
            print(f"\n[{idx+1}/{len(configs)}] {cfg_path.name}")
            _, ok, msg = render_config(cfg_path, args.output_dir, skip_existing)
            print(f"  {msg}")
            if ok:
                succeeded += 1
            else:
                failed += 1
                errors.append((cfg_path.name, msg))
                if args.fail_fast:
                    print("[ABORT] --fail_fast set, stopping.")
                    break
    else:
        print(f"[INFO] Parallel jobs: {args.jobs}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(render_config, cfg, args.output_dir, skip_existing): cfg
                for cfg in configs
            }
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                cfg_path = futures[future]
                _, ok, msg = future.result()
                status = "✓" if ok else "✗"
                print(f"  [{idx+1}/{len(configs)}] {status} {cfg_path.name}: {msg}")
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    errors.append((cfg_path.name, msg))
                    if args.fail_fast:
                        ex.shutdown(wait=False, cancel_futures=True)
                        break

    print(f"\n{'─'*50}")
    print(f"[DONE] {succeeded} succeeded  |  {failed} failed  |  {len(configs)} total")
    if errors:
        print("\nFailed configs:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()