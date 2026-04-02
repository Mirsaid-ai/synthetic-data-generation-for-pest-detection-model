"""Pipeline orchestrator — runs in system Python (no bpy).

Performs depth/normal analysis and assembles rendered frames into an MP4.
"""

import json
import os
import random
import shutil
import subprocess
import tempfile
import time as _time
import uuid
from collections import Counter

import cv2
import numpy as np
from generator.config import (
    FRAMES_DIR,
    LABELS_DIR,
    NUM_FRAMES,
    PEST_FORWARD_AXIS,
    PEST_PARAMS,
    PEST_REAL_SIZES_M,
    PEST_TYPES,
    PLANE_HEIGHT,
    PLANE_WIDTH,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    SPRITES_DIR,
    VIDEOS_DIR,
    FPS,
)
from concurrent.futures import ThreadPoolExecutor

from generator.compositing import composite_frames

from generator.depth_estimator import (
    build_movement_mask,
    build_surface_group_masks,
    compute_inference_strategy,
    estimate_gravity,
    estimate_metric3d,
    save_depth_preview,
    save_gravity_preview,
    save_movement_mask_preview,
    save_surface_preview,
    sample_pest_positions_from_probability,
)

SURFACE_GROUPS = ("up", "side_left", "side_right", "side_toward", "down")
DEFAULT_SPAWN_PROBS = {
    "up": 0.78,
    "side_left": 0.067,
    "side_right": 0.067,
    "side_toward": 0.066,
    "down": 0.02,
}
GLOBAL_PEST_SIZE_MULTIPLIER = 1.8

# Pest count sampling for each video:
#   P(N=0) = 0.25 (null cases)
#   P(N=1) = 0.30
#   Remaining 0.45 distributed exponentially over N=2..6.
NULL_CASE_PROB = 0.25
ONE_PEST_PROB = 0.30
MAX_PESTS_PER_VIDEO = 6
COUNT_EXP_DECAY_RATIO = 0.5  # each step has ~half the mass of the previous

# Pest type sampling for each pest slot.
PEST_TYPE_SAMPLING_WEIGHTS = {
    "cockroach": 0.50,
    "mouse": 0.30,
    "rat": 0.20,
}


def _build_num_pest_distribution():
    options = [0, 1]
    weights = [float(NULL_CASE_PROB), float(ONE_PEST_PROB)]

    tail_count = max(0, int(MAX_PESTS_PER_VIDEO) - 1)  # N=2..MAX
    tail_mass = max(0.0, 1.0 - weights[0] - weights[1])
    if tail_count > 0:
        raw = [float(COUNT_EXP_DECAY_RATIO) ** i for i in range(tail_count)]
        raw_sum = sum(raw)
        if raw_sum <= 1e-12:
            tail = [tail_mass / float(tail_count)] * tail_count
        else:
            tail = [tail_mass * (v / raw_sum) for v in raw]
        options.extend(range(2, int(MAX_PESTS_PER_VIDEO) + 1))
        weights.extend(tail)

    total = sum(weights)
    if total <= 1e-12:
        return [0], [1.0]
    weights = [w / total for w in weights]
    return options, weights


NUM_PEST_OPTIONS, NUM_PEST_WEIGHTS = _build_num_pest_distribution()
PEST_TYPE_OPTIONS = [p for p in PEST_TYPES if p in PEST_TYPE_SAMPLING_WEIGHTS]
if not PEST_TYPE_OPTIONS:
    PEST_TYPE_OPTIONS = list(PEST_TYPES)
PEST_TYPE_WEIGHTS = [float(PEST_TYPE_SAMPLING_WEIGHTS.get(p, 1.0)) for p in PEST_TYPE_OPTIONS]


def generate_video(
    image_path,
    video_id=None,
    job_id=None,
    frames_root=None,
    labels_root=None,
    videos_root=None,
    num_frames=None,
    fps=None,
    assemble_video=True,
    frame_format="png",
    save_scene_previews=True,
    save_mask_previews=True,
    save_movement_masks=True,
    keep_only_frame_outputs=False,
    save_every_n=1,
    keep_full_annotations=False,
):
    """Run the full generation pipeline for one kitchen image.

    Args:
        image_path: Absolute path to the uploaded kitchen image.
        video_id: Optional video identifier. Generated if not provided.
        job_id: Legacy alias for video_id (kept for backward compatibility).
        frames_root: Override for frames output directory.
        labels_root: Override for labels output directory.
        videos_root: Override for videos output directory.
        num_frames: Optional override for number of frames to render.
        fps: Optional override for video/compositing frame rate.
        assemble_video: Whether to assemble frame PNGs into an MP4.
        frame_format: Frame image format (png, jpg/jpeg, webp).
        save_scene_previews: Whether to save depth/surface/gravity preview images.
        save_mask_previews: Whether to save per-frame pest mask preview images.
        save_movement_masks: Whether to save per-surface movement mask PNGs.
        keep_only_frame_outputs: If True, delete non-frame artifacts from frames_dir
            after generation, keeping only frame_*.{png,jpg,jpeg,webp}.
        save_every_n: Persist frame 1 and then every Nth rendered frame while
            still simulating all frames. Saved frame numbering matches the
            original timeline.
        keep_full_annotations: If True, keep COCO entries for all simulated
            frames even when only a sparse subset of frame images is saved.

    Returns:
        Dict with video_id/job_id, video_path, frames_dir, labels_dir,
        pest_counts, fps, num_frames, and per-pest generation metadata.
    """
    def _positive_int(value, default):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _normalize_frame_format(value):
        fmt = str(value or "png").strip().lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in {"png", "jpg", "webp"}:
            fmt = "png"
        return fmt

    if video_id is None:
        video_id = job_id
    if video_id is None:
        video_id = uuid.uuid4().hex[:8]
    job_id = video_id

    frames_dir = os.path.join(frames_root or FRAMES_DIR, job_id)
    labels_dir = os.path.join(labels_root or LABELS_DIR, job_id)
    video_path = os.path.join(videos_root or VIDEOS_DIR, f"{job_id}.mp4")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    effective_fps = _positive_int(fps, FPS)
    effective_num_frames = _positive_int(num_frames, NUM_FRAMES)
    effective_save_every_n = _positive_int(save_every_n, 1)
    frame_ext = _normalize_frame_format(frame_format)

    # Choose number of pests from configured non-uniform distribution.
    num_pests = random.choices(NUM_PEST_OPTIONS, weights=NUM_PEST_WEIGHTS, k=1)[0]
    pest_configs = []
    for _ in range(num_pests):
        pest_type = random.choices(PEST_TYPE_OPTIONS, weights=PEST_TYPE_WEIGHTS, k=1)[0]
        params = dict(PEST_PARAMS[pest_type])
        # Convert tuples to lists for JSON serialization
        for key in ["body_scale", "head_scale", "head_offset", "color"]:
            if params[key] is not None:
                params[key] = list(params[key])
        # Convert world-units/second speeds to per-frame world units using the
        # effective FPS for this specific generation request.
        # This ensures max speed is physically grounded regardless of frame rate.
        base_speed_wps = float(params.pop("base_speed_wps", 0.1))
        max_speed_wps  = float(params.pop("max_speed_wps",  1.0))
        params["base_speed_wps"] = base_speed_wps
        params["max_speed_wps"]  = max_speed_wps
        params["speed"]          = base_speed_wps / effective_fps
        params["max_step_world"] = max_speed_wps / effective_fps
        pest_configs.append({"type": pest_type, "params": params})

    pest_counts = Counter(cfg["type"] for cfg in pest_configs)
    pest_generation_metadata = []

    # Depth-aware placement — Metric3D v2 yields depth + normals in one pass.
    # Gravity estimation (classical VP) is independent and can run in parallel.
    # Strategy: parallel on CPU or multi-GPU, sequential on single GPU.
    strategy = compute_inference_strategy()
    print(f"Inference strategy: {strategy}")

    if strategy == "parallel":
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_metric3d = ex.submit(estimate_metric3d, image_path)
            f_gravity  = ex.submit(estimate_gravity, image_path)
        metric3d_result = f_metric3d.result()
        gravity_result  = f_gravity.result()
    else:
        metric3d_result = estimate_metric3d(image_path)
        gravity_result  = estimate_gravity(image_path)

    depth_map         = metric3d_result["depth"]
    predicted_normals = metric3d_result["normals"]
    focal_length_px   = metric3d_result["fx"]
    img_h, img_w      = depth_map.shape

    if save_scene_previews:
        save_depth_preview(depth_map, os.path.join(frames_dir, "depth_preview.jpg"))
        save_surface_preview(predicted_normals, os.path.join(frames_dir, "surface_preview.jpg"))
        save_gravity_preview(image_path, gravity_result, os.path.join(frames_dir, "gravity_preview.jpg"))
    print(f"Gravity estimate: up={gravity_result['gravity_cam'].tolist()}  "
          f"conf={gravity_result['confidence']:.2f}")

    # Build directional surface-group probability maps once (shared across pests).
    surface_group_masks = build_surface_group_masks(predicted_normals)
    groups_for_spawn = [g for g in SURFACE_GROUPS if g in surface_group_masks]
    if not groups_for_spawn:
        groups_for_spawn = list(surface_group_masks.keys())
    _spawn_group_to_idx = {g: i for i, g in enumerate(groups_for_spawn)}
    _spawn_argmax_idx = np.argmax(
        np.stack([surface_group_masks[g] for g in groups_for_spawn], axis=0), axis=0
    )

    for pest_idx, pest_cfg in enumerate(pest_configs):
        pest_type   = pest_cfg["type"]
        params      = pest_cfg["params"]
        spawn_probs = _resolve_spawn_probs(params.get("spawn_probs"))
        stickiness  = float(params.get("surface_stickiness", 0.97))
        mask_brightness = float(params.get("movement_mask_brightness", 1.0))

        # Choose which surface group this pest spawns on (weighted random).
        groups  = groups_for_spawn
        weights = [spawn_probs.get(g, 0.0) for g in groups]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0 if g == "up" else 0.0 for g in groups]
        if sum(weights) <= 0:
            weights = [1.0 / max(len(groups), 1)] * len(groups)
        spawn_surface = random.choices(groups, weights=weights)[0]

        # Sample start position from the chosen surface group.
        # Use dominant-surface-only pixels so "up" spawns do not leak onto walls.
        spawn_map = np.array(surface_group_masks[spawn_surface], dtype=np.float32)
        spawn_idx = _spawn_group_to_idx.get(spawn_surface)
        if spawn_idx is not None:
            spawn_map = np.where(_spawn_argmax_idx == spawn_idx, spawn_map, 0.0)
            if float(spawn_map.sum()) <= 1e-8:
                spawn_map = np.array(surface_group_masks[spawn_surface], dtype=np.float32)

        start_position = sample_pest_positions_from_probability(
            spawn_map,
            n=1,
            plane_width=PLANE_WIDTH,
            plane_height=PLANE_HEIGHT,
        )[0]

        # Build and save per-surface movement masks for this pest so simulation
        # can read the exact generated masks at runtime (fully mask-driven).
        surface_mask_paths = {}
        surface_mask_arrays = {}
        for surface_name in groups:
            movement_prob = build_movement_mask(
                surface_group_masks, surface_name, stickiness
            )
            movement_prob = np.clip(movement_prob * mask_brightness, 0.0, 1.0).astype(np.float32)
            surface_mask_arrays[surface_name] = movement_prob
            if save_movement_masks:
                mask_filename = f"movement_mask_{pest_type}_{pest_idx}_{surface_name}.png"
                mask_path = os.path.abspath(os.path.join(frames_dir, mask_filename))
                save_movement_mask_preview(movement_prob, surface_name, mask_path)
                surface_mask_paths[surface_name] = mask_path

        # Keep a single default mask path (spawn surface) for fallback and compatibility.
        mask_path = surface_mask_paths.get(spawn_surface)
        if not mask_path and surface_mask_paths:
            mask_path = next(iter(surface_mask_paths.values()))
        placement_mask_array = surface_mask_arrays.get(spawn_surface)
        if placement_mask_array is None and surface_mask_arrays:
            placement_mask_array = next(iter(surface_mask_arrays.values()))

        pest_cfg["start_position"]    = [float(start_position[0]), float(start_position[1])]
        pest_cfg["placement_mask_path"] = mask_path
        pest_cfg["surface_mask_paths"] = surface_mask_paths
        pest_cfg["placement_mask_array"] = placement_mask_array
        pest_cfg["surface_mask_arrays"] = surface_mask_arrays

        # Depth-based scale: apparent pixel width = real_size_m * fx / depth_m
        # => blender_scale = real_size_m * fx * PLANE_WIDTH / (depth_m * RENDER_WIDTH)
        wx, wy = start_position
        px = int(round((wx / PLANE_WIDTH + 0.5) * (img_w - 1)))
        py = int(round((0.5 - wy / PLANE_HEIGHT) * (img_h - 1)))
        px = max(0, min(img_w - 1, px))
        py = max(0, min(img_h - 1, py))
        depth_val = float(depth_map[py, px])

        real_size      = PEST_REAL_SIZES_M[pest_type]
        default_body_x = PEST_PARAMS[pest_type]["body_scale"][0]
        if depth_val > 0.1:
            raw_scale = real_size * focal_length_px * PLANE_WIDTH / (depth_val * RENDER_WIDTH)
            base_blender_scale = float(
                max(default_body_x * 0.4, min(default_body_x * 3.0, raw_scale))
            )
        else:
            base_blender_scale = float(default_body_x)

        blender_scale = float(base_blender_scale * GLOBAL_PEST_SIZE_MULTIPLIER)

        params["blender_scale"]  = blender_scale
        params["forward_axis"]   = PEST_FORWARD_AXIS.get(pest_type, "X")
        # Pass depth info through for per-frame depth-aware speed cap in compute_walk.
        params["focal_length_px"] = float(focal_length_px)

        # Metadata for generated_state.json: size + initial placement in image coordinates.
        render_px = int(round((wx / PLANE_WIDTH + 0.5) * (RENDER_WIDTH - 1)))
        render_py = int(round((0.5 - wy / PLANE_HEIGHT) * (RENDER_HEIGHT - 1)))
        render_px = max(0, min(RENDER_WIDTH - 1, render_px))
        render_py = max(0, min(RENDER_HEIGHT - 1, render_py))
        approx_pixel_width = float(blender_scale * RENDER_WIDTH / PLANE_WIDTH)
        relative_size_scale = (
            float(blender_scale / default_body_x) if float(default_body_x) > 1e-8 else None
        )

        pest_generation_metadata.append({
            "pest_index": int(pest_idx),
            "pest_type": pest_type,
            "relative_size_scale": round(relative_size_scale, 4) if relative_size_scale is not None else None,
            "relative_size_image_fraction": round(approx_pixel_width / float(RENDER_WIDTH), 5),
            "approx_initial_pixel_width": round(approx_pixel_width, 2),
            "initial_position_image_px": {"x": int(render_px), "y": int(render_py)},
            "initial_position_image_norm": {
                "x": round(render_px / float(max(RENDER_WIDTH - 1, 1)), 6),
                "y": round(render_py / float(max(RENDER_HEIGHT - 1, 1)), 6),
            },
        })

    dense_mp4_with_sparse_persist = assemble_video and effective_save_every_n > 1

    if dense_mp4_with_sparse_persist:
        with tempfile.TemporaryDirectory(prefix=f"{job_id}_dense_", dir=os.path.dirname(frames_dir)) as temp_root:
            temp_frames_dir = os.path.join(temp_root, "frames")
            temp_labels_dir = os.path.join(temp_root, "labels")
            print(f"Running dense 2D compositing for MP4 job {job_id}...")
            _t0 = _time.monotonic()
            composite_frames(
                image_path=os.path.abspath(image_path),
                pest_configs=pest_configs,
                frames_dir=temp_frames_dir,
                labels_dir=temp_labels_dir,
                num_frames=effective_num_frames,
                sprites_dir=SPRITES_DIR,
                render_width=RENDER_WIDTH,
                render_height=RENDER_HEIGHT,
                plane_width=PLANE_WIDTH,
                plane_height=PLANE_HEIGHT,
                depth_map=depth_map,
                fps=effective_fps,
                surface_group_masks=surface_group_masks,
                normals=predicted_normals,
                save_mask_previews=bool(save_mask_previews),
                frame_format=frame_ext,
                save_every_n=1,
                keep_full_annotations=True,
            )
            print(f"Dense compositing finished in {_time.monotonic() - _t0:.1f}s (job {job_id})")
            _assemble_video(temp_frames_dir, video_path, fps=effective_fps, frame_ext=frame_ext)
            _persist_sparse_outputs_from_dense(
                dense_frames_dir=temp_frames_dir,
                dense_labels_dir=temp_labels_dir,
                sparse_frames_dir=frames_dir,
                sparse_labels_dir=labels_dir,
            )
    else:
        print(f"Running 2D compositing for job {job_id}...")
        _t0 = _time.monotonic()
        composite_frames(
            image_path=os.path.abspath(image_path),
            pest_configs=pest_configs,
            frames_dir=frames_dir,
            labels_dir=labels_dir,
            num_frames=effective_num_frames,
            sprites_dir=SPRITES_DIR,
            render_width=RENDER_WIDTH,
            render_height=RENDER_HEIGHT,
            plane_width=PLANE_WIDTH,
            plane_height=PLANE_HEIGHT,
            depth_map=depth_map,
            fps=effective_fps,
            surface_group_masks=surface_group_masks,
            normals=predicted_normals,
            save_mask_previews=bool(save_mask_previews),
            frame_format=frame_ext,
            save_every_n=effective_save_every_n,
            keep_full_annotations=bool(keep_full_annotations),
        )
        print(f"Compositing finished in {_time.monotonic() - _t0:.1f}s (job {job_id})")

        # Assemble frames into MP4 (optional for training-only generation).
        if assemble_video:
            video_fps = max(float(effective_fps) / float(effective_save_every_n), 1.0)
            _assemble_video(frames_dir, video_path, fps=video_fps, frame_ext=frame_ext)
        else:
            video_path = None

    if not assemble_video:
        video_path = None

    if keep_only_frame_outputs:
        _prune_auxiliary_frame_files(frames_dir)

    return {
        "video_id": job_id,
        "job_id": job_id,
        "video_path": video_path,
        "frames_dir": frames_dir,
        "labels_dir": labels_dir,
        "pest_counts": {
            "mouse": int(pest_counts.get("mouse", 0)),
            "rat": int(pest_counts.get("rat", 0)),
            "cockroach": int(pest_counts.get("cockroach", 0)),
        },
        "pest_size_multiplier": float(GLOBAL_PEST_SIZE_MULTIPLIER),
        "pest_generation_metadata": pest_generation_metadata,
        "fps": effective_fps,
        "num_frames": effective_num_frames,
        "saved_num_frames": (
            effective_num_frames
            if effective_save_every_n <= 1
            else 1 + (effective_num_frames // effective_save_every_n)
        ),
        "save_every_n": effective_save_every_n,
    }


def _resolve_spawn_probs(spawn_probs):
    """Resolve spawn probabilities with backward compatibility for legacy 'side'."""
    if not isinstance(spawn_probs, dict):
        return dict(DEFAULT_SPAWN_PROBS)

    probs = {g: max(float(spawn_probs.get(g, 0.0)), 0.0) for g in SURFACE_GROUPS}
    side_groups = ("side_left", "side_right", "side_toward")
    side_total = sum(probs[g] for g in side_groups)
    legacy_side = max(float(spawn_probs.get("side", 0.0)), 0.0)

    if side_total <= 1e-8 and legacy_side > 0.0:
        split = legacy_side / len(side_groups)
        for g in side_groups:
            probs[g] = split

    if sum(probs.values()) <= 1e-8:
        return dict(DEFAULT_SPAWN_PROBS)
    return probs


def _persist_sparse_outputs_from_dense(
    dense_frames_dir,
    dense_labels_dir,
    sparse_frames_dir,
    sparse_labels_dir,
):
    os.makedirs(sparse_frames_dir, exist_ok=True)
    os.makedirs(sparse_labels_dir, exist_ok=True)

    dense_ann_path = os.path.join(dense_labels_dir, "annotations.json")
    with open(dense_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    for img in coco.get("images", []):
        fname = img.get("file_name")
        if not fname:
            continue
        src = os.path.join(dense_frames_dir, fname)
        dst = os.path.join(sparse_frames_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    with open(os.path.join(sparse_labels_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "images": coco.get("images", []),
                "annotations": coco.get("annotations", []),
                "categories": coco.get("categories", []),
            },
            f,
            indent=2,
        )


def _prune_auxiliary_frame_files(frames_dir):
    """Keep only rendered frame image files in frames_dir, remove artifacts."""
    try:
        names = os.listdir(frames_dir)
    except OSError:
        return

    for name in names:
        path = os.path.join(frames_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        is_frame = (
            lower.startswith("frame_")
            and lower.endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        if is_frame:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _assemble_video(frames_dir, video_path, fps=FPS, frame_ext="png"):
    """Combine rendered frames into an H.264 MP4 using ffmpeg (browser-compatible).

    Falls back to OpenCV (mp4v) if ffmpeg is not available.

    Args:
        frames_dir: Directory containing frame_XXXX.<ext> files.
        video_path: Output MP4 path.
        fps: Frame rate for video encoding.
        frame_ext: Frame file extension (png, jpg/jpeg, webp).
    """
    frame_ext = str(frame_ext or "png").strip().lower()
    if frame_ext == "jpeg":
        frame_ext = "jpg"
    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.startswith("frame_") and f.lower().endswith(f".{frame_ext}")
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    if ffmpeg:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as manifest:
            manifest_path = manifest.name
            for fname in frame_files:
                frame_path = os.path.join(frames_dir, fname).replace("'", "'\\''")
                manifest.write(f"file '{frame_path}'\n")
        cmd = [
            ffmpeg, "-y",
            "-r", str(fps),
            "-f", "concat",
            "-safe", "0",
            "-i", manifest_path,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            os.remove(manifest_path)
        except OSError:
            pass
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        print(f"Video assembled (H.264): {video_path} ({len(frame_files)} frames)")
        return

    # Fallback: OpenCV with mp4v (may not play in all browsers)
    print("WARNING: ffmpeg not found, falling back to OpenCV mp4v codec.")
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for fname in frame_files:
        writer.write(cv2.imread(os.path.join(frames_dir, fname)))
    writer.release()
    print(f"Video assembled (mp4v): {video_path} ({len(frame_files)} frames)")
