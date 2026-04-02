"""Flask web application — upload kitchen images and generate synthetic pest videos."""

# Load .env at startup so GEMINI_API_KEY (and any other vars) are available
from dotenv import load_dotenv
load_dotenv()

import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify

from generator.config import (
    OUTPUT_DIR as PROJECT_OUTPUT_DIR,
    UPLOAD_DIR as DEFAULT_UPLOAD_DIR,
    FRAMES_DIR as DEFAULT_FRAMES_DIR,
    VIDEOS_DIR as DEFAULT_VIDEOS_DIR,
    LABELS_DIR as DEFAULT_LABELS_DIR,
)
from generator.depth_estimator import (
    preload_models,
    estimate_metric3d,
    save_depth_preview,
    save_surface_preview,
)
from generator.kitchen_img.download_kitchens import (
    DATASET_SPLIT as DOWNLOAD_SPLIT,
    OUT_DIR as UNCURATED_IMG_DIR,
    TARGET as DEFAULT_DOWNLOAD_TARGET,
    extract_places_filename,
    link_kitchen_to_places,
    list_local_kitchen_images,
    mark_places_as_seen,
)
from generator.pipeline import generate_video


# Test tab roots (can be redirected to temp when TESTING_MODE=True)
UPLOAD_DIR = DEFAULT_UPLOAD_DIR
FRAMES_DIR = DEFAULT_FRAMES_DIR
VIDEOS_DIR = DEFAULT_VIDEOS_DIR
LABELS_DIR = DEFAULT_LABELS_DIR

# Real tab roots (always persistent; kept training-compatible)
REAL_TRAIN_FRAME_STRIDE = 10


# --- Testing mode: use temp directory instead of outputs/ ---
# Set to False (or remove this block) when you want to save permanently
TESTING_MODE = True

if TESTING_MODE:
    import tempfile
    _TEMP_DIR = tempfile.mkdtemp(prefix="pest_gen_")
    UPLOAD_DIR = os.path.join(_TEMP_DIR, "uploads")
    FRAMES_DIR = os.path.join(_TEMP_DIR, "frames")
    VIDEOS_DIR = os.path.join(_TEMP_DIR, "videos")
    LABELS_DIR = os.path.join(_TEMP_DIR, "labels")
    print(f"TESTING MODE: outputs go to {_TEMP_DIR} (auto-deleted on exit)")
# ---------------------------------------------------------

app = Flask(__name__)
app.secret_key = "synthetic-pest-gen-dev-key"

# Kitchen image directories:
#   generator/kitchen_img/uncurated_img  (downloaded + generated, pending review)
#   generator/kitchen_img/curated_img    (approved images)
KITCHEN_ROOT_DIR = os.path.dirname(UNCURATED_IMG_DIR)
CURATED_IMG_DIR = os.path.join(KITCHEN_ROOT_DIR, "curated_img")
TRAIN_TEST_SPLIT_PATH = os.path.join(KITCHEN_ROOT_DIR, "test_train_split.csv")

# Ensure output directories exist
for d in [
    UPLOAD_DIR,
    FRAMES_DIR,
    VIDEOS_DIR,
    LABELS_DIR,
    UNCURATED_IMG_DIR,
    CURATED_IMG_DIR,
]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
CURATOR_PREVIEW_DIR = os.path.join(KITCHEN_ROOT_DIR, ".curator_cache")
CURATED_PAGE_SIZE = 5
GENERATED_STATE_PATH = os.path.join(PROJECT_OUTPUT_DIR, "generated_state.json")
os.makedirs(CURATOR_PREVIEW_DIR, exist_ok=True)


# Stores total render time (seconds) keyed by job_id
_render_times = {}
# Stores the source image path used for each job so regeneration doesn't need a re-upload
_source_images = {}
# Stores effective fps keyed by job_id so result playback overlays stay in sync
_job_fps = {}
# Tracks background download status
_download_status = {
    "running": False,
    "message": "",
    "phase": "idle",
    "saved": 0,
    "total": 0,
    "current_file": None,
}
_download_status_lock = threading.Lock()
_real_state_lock = threading.Lock()
_real_batches = {}
_real_batches_lock = threading.Lock()
REAL_BATCH_MAX_TRACKED = 25
_kitchen_id_lock = threading.Lock()
_kitchen_id_counter = 0
KITCHEN_NAME_RE = re.compile(r"^kitchen_(\d+)\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)
LEGACY_KITCHEN_IMG_NAME_RE = re.compile(
    r"^kitchen_img_(\d+)\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)


def _background_warm_models():
    start = time.time()
    try:
        print("Metric3D model warmup started...")
        preload_models()
        print(f"Metric3D model warmup complete in {time.time() - start:.1f}s")
    except Exception as e:
        print(f"WARNING: Metric3D model warmup failed: {e}")


# Warm the model in the background so the first generation request is faster.
threading.Thread(target=_background_warm_models, daemon=True).start()


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _parse_positive_int(value, default=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(value, default=1.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def list_curated_images():
    """Return image files in CURATED_IMG_DIR (images approved by user)."""
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for name in sorted(os.listdir(CURATED_IMG_DIR)):
        path = os.path.join(CURATED_IMG_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(image_exts):
            files.append(name)
    return files


def _get_curated_page(page, page_size=CURATED_PAGE_SIZE):
    images = list_curated_images()
    total = len(images)
    if total == 0:
        return {
            "images": [],
            "total": 0,
            "page": 1,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False,
            "prev_page": 1,
            "next_page": 1,
        }

    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    page_images = images[start:start + page_size]
    return {
        "images": page_images,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else 1,
        "next_page": page + 1 if page < total_pages else total_pages,
    }


def _generate_video_for_image(image_path):
    t0 = time.time()
    result = generate_video(
        image_path,
        frames_root=FRAMES_DIR,
        labels_root=LABELS_DIR,
        videos_root=VIDEOS_DIR,
    )
    rid = result.get("video_id") or result.get("job_id")
    if rid:
        _render_times[rid] = round(time.time() - t0, 1)
        _source_images[rid] = image_path
        try:
            _job_fps[rid] = int(result.get("fps", 10))
        except (TypeError, ValueError):
            _job_fps[rid] = 10
    return result


def _generate_video_for_image_with_params(
    image_path,
    num_frames,
    fps,
    use_real_outputs=False,
    assemble_video=True,
    frames_root=None,
    labels_root=None,
    videos_root=None,
    save_every_n=None,
):
    t0 = time.time()
    # Real generator outputs must be explicit split roots (outputs/train|test/...).
    # This prevents accidental fallback to legacy outputs/{frames,labels,videos}.
    if use_real_outputs and (not frames_root or not labels_root or not videos_root):
        raise ValueError(
            "Real generation requires explicit split output roots "
            "(frames_root, labels_root, videos_root)."
        )

    resolved_frames_root = frames_root or FRAMES_DIR
    resolved_labels_root = labels_root or LABELS_DIR
    resolved_videos_root = videos_root or VIDEOS_DIR
    resolved_save_every_n = save_every_n or (REAL_TRAIN_FRAME_STRIDE if use_real_outputs else 1)
    result = generate_video(
        image_path,
        frames_root=resolved_frames_root,
        labels_root=resolved_labels_root,
        videos_root=resolved_videos_root,
        num_frames=num_frames,
        fps=fps,
        assemble_video=assemble_video,
        frame_format="png",
        save_scene_previews=not use_real_outputs,
        save_mask_previews=not use_real_outputs,
        save_movement_masks=not use_real_outputs,
        keep_only_frame_outputs=use_real_outputs,
        save_every_n=resolved_save_every_n,
        keep_full_annotations=use_real_outputs,
    )
    rid = result.get("video_id") or result.get("job_id")
    if rid:
        _render_times[rid] = round(time.time() - t0, 1)
        _source_images[rid] = image_path
        try:
            _job_fps[rid] = int(result.get("fps", fps))
        except (TypeError, ValueError):
            _job_fps[rid] = int(fps)
    return result


def _render_generate_page(job_context=None):
    page = _parse_positive_int(request.args.get("page", 1), default=1)
    test_length_seconds = _parse_positive_float(request.args.get("length_seconds"), default=24.0)
    test_fps = _parse_positive_int(request.args.get("fps"), default=10)
    curated = _get_curated_page(page)
    context = {
        "active_tab": "generate",
        "test_form_values": {
            "length_seconds": round(test_length_seconds, 2),
            "fps": test_fps,
        },
        "curated_page_images": curated["images"],
        "curated_page": curated["page"],
        "curated_total": curated["total"],
        "curated_total_pages": curated["total_pages"],
        "curated_has_prev": curated["has_prev"],
        "curated_has_next": curated["has_next"],
        "curated_prev_page": curated["prev_page"],
        "curated_next_page": curated["next_page"],
    }
    if job_context:
        context.update(job_context)
    return render_template("index.html", **context)


def _render_real_generate_page(job_context=None, page_override=None):
    page = page_override
    if page is None:
        page = _parse_positive_int(request.args.get("rpage", 1), default=1)
    page = _parse_positive_int(page, default=1)
    curated = _get_curated_page(page)
    context = {
        "active_tab": "real_generator",
        "real_curated_page_images": curated["images"],
        "real_curated_page": curated["page"],
        "real_curated_total": curated["total"],
        "real_curated_total_pages": curated["total_pages"],
        "real_curated_has_prev": curated["has_prev"],
        "real_curated_has_next": curated["has_next"],
        "real_curated_prev_page": curated["prev_page"],
        "real_curated_next_page": curated["next_page"],
    }
    if job_context:
        context.update(job_context)
    return render_template("index.html", **context)


def _init_kitchen_id_counter():
    global _kitchen_id_counter
    max_id = 0
    for name in list_curated_images():
        m = KITCHEN_NAME_RE.match(name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    _kitchen_id_counter = max_id


def _allocate_kitchen_filename(ext):
    ext = (ext or "").lower().strip()
    if not ext.startswith("."):
        ext = f".{ext}" if ext else ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        ext = ".jpg"

    global _kitchen_id_counter
    with _kitchen_id_lock:
        if _kitchen_id_counter <= 0:
            _init_kitchen_id_counter()
        while True:
            _kitchen_id_counter += 1
            candidate = f"kitchen_{_kitchen_id_counter:04d}{ext}"
            if not os.path.exists(os.path.join(CURATED_IMG_DIR, candidate)):
                return candidate


def _kitchen_image_id(filename):
    return os.path.basename(filename)


def _load_train_test_split():
    if not os.path.exists(TRAIN_TEST_SPLIT_PATH):
        raise FileNotFoundError(
            f"Missing split file: {TRAIN_TEST_SPLIT_PATH}. "
            "Generate it with generator/kitchen_img/test_train_split.py."
        )

    split_map = {}
    with open(TRAIN_TEST_SPLIT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "train"]:
            raise ValueError(
                f"{TRAIN_TEST_SPLIT_PATH} must have exactly these columns: id, train"
            )
        for row in reader:
            kitchen_id = (row.get("id") or "").strip()
            train_flag = (row.get("train") or "").strip()
            if not kitchen_id:
                continue
            if train_flag not in {"0", "1"}:
                raise ValueError(
                    f"Invalid train flag for {kitchen_id!r}: {train_flag!r}. Expected 0 or 1."
                )
            split_map[kitchen_id] = "train" if train_flag == "1" else "test"
    return split_map


def _real_output_roots_for_split(split_name):
    split_root = os.path.join(PROJECT_OUTPUT_DIR, split_name)
    return {
        "frames_root": os.path.join(split_root, "frames"),
        "labels_root": os.path.join(split_root, "labels"),
        "videos_root": os.path.join(split_root, "videos"),
    }


def _real_worker_count(target_count):
    cpu = max(1, os.cpu_count() or 1)
    return max(1, min(target_count, cpu, 4))


def _load_generated_state():
    if not os.path.exists(GENERATED_STATE_PATH):
        return {"generated_videos": []}
    try:
        with open(GENERATED_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_videos": []}

    def _normalize_generated_row(row):
        if not isinstance(row, dict):
            return row
        out = dict(row)
        if "kitchen_id" not in out:
            if "kitchen_img_id" in out:
                out["kitchen_id"] = out.get("kitchen_img_id")
            elif "kitchen_image_id" in out:
                out["kitchen_id"] = out.get("kitchen_image_id")
        if isinstance(out.get("kitchen_id"), str) and out["kitchen_id"].startswith("kitchen_img_"):
            out["kitchen_id"] = "kitchen_" + out["kitchen_id"][len("kitchen_img_"):]
        if "video_id" not in out:
            out["video_id"] = out.get("job_id")
        out.pop("kitchen_img_id", None)
        out.pop("kitchen_image_id", None)
        out.pop("kitchen_filename", None)
        return out

    if isinstance(data, list):
        return {"generated_videos": [_normalize_generated_row(r) for r in data]}
    if isinstance(data, dict):
        items = data.get("generated_videos", [])
        if not isinstance(items, list):
            items = []
        return {"generated_videos": [_normalize_generated_row(r) for r in items]}
    return {"generated_videos": []}


def _save_generated_state(state):
    payload = {
        "generated_videos": list(state.get("generated_videos", [])),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(GENERATED_STATE_PATH), exist_ok=True)
    with open(GENERATED_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_generated_state_rows(rows):
    if not rows:
        return

    with _real_state_lock:
        state = _load_generated_state()
        current = state.get("generated_videos", [])
        current.extend(rows)
        state["generated_videos"] = current
        _save_generated_state(state)


def _prune_real_batches_locked():
    if len(_real_batches) <= REAL_BATCH_MAX_TRACKED:
        return
    done_ids = sorted(
        (
            (bid, batch.get("created_ts", 0.0))
            for bid, batch in _real_batches.items()
            if batch.get("done")
        ),
        key=lambda x: x[1],
    )
    for bid, _ in done_ids:
        if len(_real_batches) <= REAL_BATCH_MAX_TRACKED:
            break
        _real_batches.pop(bid, None)


def _real_batch_payload(batch_id, batch):
    requested = max(0, int(batch.get("requested", 0)))
    completed = max(0, int(batch.get("completed", 0)))
    generated = max(0, int(batch.get("generated", 0)))
    progress_pct = int(round((completed / requested) * 100.0)) if requested else 0
    payload = {
        "batch_id": batch_id,
        "requested": requested,
        "completed": min(completed, requested) if requested else completed,
        "generated": generated,
        "failed": max(0, completed - generated),
        "workers": max(1, int(batch.get("workers", 1))),
        "running": not bool(batch.get("done")),
        "done": bool(batch.get("done")),
        "progress_pct": max(0, min(100, progress_pct)),
        "batch_time": batch.get("batch_time"),
        "state_path": GENERATED_STATE_PATH,
        "generate_mp4": bool(batch.get("generate_mp4")),
        "error": batch.get("error"),
        "results": list(batch.get("results", [])),
        "failures": list(batch.get("failures", [])),
        "form_values": dict(batch.get("form_values", {})),
    }
    return payload


def _get_real_batch_snapshot(batch_id):
    with _real_batches_lock:
        batch = _real_batches.get(batch_id)
        if not batch:
            return None
        return _real_batch_payload(batch_id, batch)


def _run_real_generation_batch(
    batch_id,
    selected_filenames,
    split_map,
    length_seconds,
    fps,
    num_frames,
    workers,
    generate_mp4,
):
    def _ui_row(row):
        return {
            "job_id": row["job_id"],
            "video_id": row["video_id"],
            "kitchen_id": row["kitchen_id"],
            "split": row["split"],
            "length_seconds": row["length_of_video_seconds"],
            "fps": row["fps"],
            "mouse_count": row["mouse_count"],
            "rat_count": row["rat_count"],
            "cockroach_count": row["cockroach_count"],
            "render_time": row["time_taken_to_generate_seconds"],
        }

    def _run_single(curated_filename):
        image_path = os.path.join(CURATED_IMG_DIR, curated_filename)
        kitchen_id = _kitchen_image_id(curated_filename)
        split_name = split_map.get(kitchen_id)
        if split_name not in {"train", "test"}:
            raise ValueError(
                f"Kitchen {kitchen_id} is missing from {TRAIN_TEST_SPLIT_PATH}. "
                "Regenerate the split file before running the real generator."
            )
        output_roots = _real_output_roots_for_split(split_name)
        t0 = time.time()
        result = _generate_video_for_image_with_params(
            image_path,
            num_frames=num_frames,
            fps=fps,
            use_real_outputs=True,
            assemble_video=generate_mp4,
            frames_root=output_roots["frames_root"],
            labels_root=output_roots["labels_root"],
            videos_root=output_roots["videos_root"],
        )
        elapsed = round(time.time() - t0, 2)
        generated_at = datetime.now().isoformat(timespec="seconds")
        pest_counts = result.get("pest_counts") or {}
        pest_generation_metadata = result.get("pest_generation_metadata") or []
        video_id = result.get("video_id") or result.get("job_id")
        return {
            "job_id": video_id,
            "video_id": video_id,
            "kitchen_id": kitchen_id,
            "split": split_name,
            "length_of_video_seconds": round(length_seconds, 2),
            "fps": fps,
            "mouse_count": int(pest_counts.get("mouse", 0)),
            "rat_count": int(pest_counts.get("rat", 0)),
            "cockroach_count": int(pest_counts.get("cockroach", 0)),
            "date_time_generated": generated_at,
            "time_taken_to_generate_seconds": elapsed,
            "pest_size_multiplier": float(result.get("pest_size_multiplier", 1.0)),
            "pest_generation_metadata": pest_generation_metadata,
            "frames_dir": os.path.join("outputs", split_name, "frames", video_id),
            "labels_dir": os.path.join("outputs", split_name, "labels", video_id),
            "video_path": result.get("video_path"),
        }

    started_batch = time.time()
    rows = []
    results_for_ui = []
    failures = []

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            future_map = {
                pool_exec.submit(_run_single, filename): filename
                for filename in selected_filenames
            }
            for future in as_completed(future_map):
                filename = future_map[future]
                try:
                    row = future.result()
                    rows.append(row)
                    results_for_ui.append(_ui_row(row))
                    # Persist each completed job immediately so generated_state.json
                    # is always up to date while a batch is still running.
                    try:
                        _append_generated_state_rows([row])
                    except Exception as e:
                        failures.append(f"{filename}: failed writing generated_state.json: {e}")
                except Exception as e:
                    failures.append(f"{filename}: {e}")
                finally:
                    with _real_batches_lock:
                        batch = _real_batches.get(batch_id)
                        if batch:
                            batch["completed"] = int(batch.get("completed", 0)) + 1
                            batch["generated"] = len(rows)
                            batch["failures"] = list(failures)
                            batch["results"] = list(results_for_ui)

        rows.sort(key=lambda r: r["job_id"])
        results_for_ui.sort(key=lambda r: r["job_id"])
        batch_time = round(time.time() - started_batch, 2)

        with _real_batches_lock:
            batch = _real_batches.get(batch_id)
            if batch:
                batch.update({
                    "done": True,
                    "completed": int(batch.get("requested", len(selected_filenames))),
                    "generated": len(rows),
                    "results": results_for_ui,
                    "failures": list(failures),
                    "batch_time": batch_time,
                    "error": None,
                })
    except Exception as e:
        with _real_batches_lock:
            batch = _real_batches.get(batch_id)
            if batch:
                batch.update({
                    "done": True,
                    "completed": int(batch.get("requested", len(selected_filenames))),
                    "results": [],
                    "failures": list(failures),
                    "batch_time": round(time.time() - started_batch, 2),
                    "error": str(e),
                })


def _curator_preview_names(filename):
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(filename)[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60]
    base = f"{safe_stem}_{digest}"
    return f"{base}_depth.jpg", f"{base}_surface.jpg"


def _ensure_scene_analysis_for_path(source_path):
    """Run depth/surface analysis for any image path, caching in CURATOR_PREVIEW_DIR."""
    if not os.path.exists(source_path):
        return None, None

    filename = os.path.basename(source_path)
    depth_name, surface_name = _curator_preview_names(filename)
    depth_path = os.path.join(CURATOR_PREVIEW_DIR, depth_name)
    surface_path = os.path.join(CURATOR_PREVIEW_DIR, surface_name)

    source_mtime = os.path.getmtime(source_path)
    cache_fresh = (
        os.path.exists(depth_path)
        and os.path.exists(surface_path)
        and os.path.getmtime(depth_path) >= source_mtime
        and os.path.getmtime(surface_path) >= source_mtime
    )

    if not cache_fresh:
        metric3d_result = estimate_metric3d(source_path)
        save_depth_preview(metric3d_result["depth"], depth_path)
        save_surface_preview(metric3d_result["normals"], surface_path)

    return depth_name, surface_name


def _ensure_curator_scene_analysis(filename):
    source_path = os.path.join(UNCURATED_IMG_DIR, filename)
    return _ensure_scene_analysis_for_path(source_path)


def _migrate_curated_images_to_kitchen_ids():
    renamed = 0
    for name in list_curated_images():
        if KITCHEN_NAME_RE.match(name):
            continue
        src = os.path.join(CURATED_IMG_DIR, name)
        ext = os.path.splitext(name)[1].lower() or ".jpg"
        legacy_m = LEGACY_KITCHEN_IMG_NAME_RE.match(name)
        if legacy_m:
            new_name = f"kitchen_{int(legacy_m.group(1)):04d}{ext}"
            if os.path.exists(os.path.join(CURATED_IMG_DIR, new_name)):
                new_name = _allocate_kitchen_filename(ext)
        else:
            new_name = _allocate_kitchen_filename(ext)
        dst = os.path.join(CURATED_IMG_DIR, new_name)
        try:
            shutil.move(src, dst)
            renamed += 1
        except OSError as e:
            print(f"WARNING: failed to rename curated image {name} -> {new_name}: {e}")
            continue

        places_name = extract_places_filename(name)
        if places_name:
            mark_places_as_seen({places_name})
            link_kitchen_to_places(new_name, places_name)

    if renamed:
        print(f"Migrated {renamed} curated image(s) to kitchen_id naming.")
    _init_kitchen_id_counter()


_migrate_curated_images_to_kitchen_ids()


# ---------------------------------------------------------------------------
# Generator tab
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return _render_generate_page()


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        flash("Please upload a valid image file (PNG, JPG, BMP, WEBP).")
        return redirect(url_for("index"))

    page = _parse_positive_int(request.form.get("page", 1), default=1)
    length_seconds = _parse_positive_float(request.form.get("length_seconds"), default=24.0)
    fps = _parse_positive_int(request.form.get("fps"), default=10)
    num_frames = max(1, int(round(length_seconds * fps)))

    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        result = _generate_video_for_image_with_params(
            save_path,
            num_frames=num_frames,
            fps=fps,
            use_real_outputs=False,
            assemble_video=True,
        )
    except Exception as e:
        flash(f"Generation failed: {e}")
        return redirect(url_for("index", page=page, length_seconds=length_seconds, fps=fps))

    rid = result.get("video_id") or result.get("job_id")
    return redirect(url_for("results", job_id=rid, page=page, length_seconds=length_seconds, fps=fps))


@app.route("/generate/curated", methods=["POST"])
def generate_from_curated():
    filename = (request.form.get("filename") or "").strip()
    page = _parse_positive_int(request.form.get("page", 1), default=1)
    length_seconds = _parse_positive_float(request.form.get("length_seconds"), default=24.0)
    fps = _parse_positive_int(request.form.get("fps"), default=10)
    num_frames = max(1, int(round(length_seconds * fps)))

    images = list_curated_images()
    if not filename or filename not in images:
        flash("Selected curated image was not found.")
        return redirect(url_for("index", page=page, length_seconds=length_seconds, fps=fps))

    image_path = os.path.join(CURATED_IMG_DIR, filename)
    if not os.path.exists(image_path):
        flash(f"Image file not found: {filename}")
        return redirect(url_for("index", page=page, length_seconds=length_seconds, fps=fps))

    try:
        result = _generate_video_for_image_with_params(
            image_path,
            num_frames=num_frames,
            fps=fps,
            use_real_outputs=False,
            assemble_video=True,
        )
    except Exception as e:
        flash(f"Generation failed for {filename}: {e}")
        return redirect(url_for("index", page=page, length_seconds=length_seconds, fps=fps))

    rid = result.get("video_id") or result.get("job_id")
    return redirect(url_for("results", job_id=rid, page=page, length_seconds=length_seconds, fps=fps))


@app.route("/results/<job_id>")
def results(job_id):
    video_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")
    frames_dir = os.path.join(FRAMES_DIR, job_id)
    labels_dir = os.path.join(LABELS_DIR, job_id)
    depth_preview_path = os.path.join(frames_dir, "depth_preview.jpg")
    surface_preview_path = os.path.join(frames_dir, "surface_preview.jpg")
    gravity_preview_path = os.path.join(frames_dir, "gravity_preview.jpg")

    if not os.path.exists(video_path):
        flash("Job not found.")
        return redirect(url_for("index"))

    all_job_files = sorted(os.listdir(frames_dir))
    frames = sorted(
        f for f in all_job_files if f.startswith("frame_") and f.endswith(".png")
    )

    _pest_mask_re = re.compile(r"^mask_preview_(pest(\d+)_(\w+))_0001\.png$")
    pest_mask_templates = []
    for fname in all_job_files:
        m = _pest_mask_re.match(fname)
        if m:
            base, pest_idx, pest_type = m.group(1), int(m.group(2)), m.group(3)
            pest_mask_templates.append({
                "pest_idx": pest_idx,
                "label": f"{pest_type.capitalize()} #{pest_idx}",
                "url_template": url_for(
                    "serve_frame", job_id=job_id,
                    filename=f"mask_preview_{base}_NNNN.png"
                ),
            })
    pest_mask_templates.sort(key=lambda t: t["pest_idx"])

    coco_data = {}
    ann_path = os.path.join(labels_dir, "annotations.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            coco_data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
    labels_by_frame = {}
    for ann in coco_data.get("annotations", []):
        fname = id_to_filename.get(ann["image_id"])
        if fname:
            labels_by_frame.setdefault(fname, []).append(ann)

    categories_by_id = {
        str(cat["id"]): cat["name"] for cat in coco_data.get("categories", [])
    }

    render_width = 640
    render_height = 480
    if coco_data.get("images"):
        render_width = coco_data["images"][0].get("width", 640)
        render_height = coco_data["images"][0].get("height", 480)

    return _render_generate_page({
        "job_id": job_id,
        "video_url": url_for("serve_video", job_id=job_id),
        "depth_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="depth_preview.jpg")
            if os.path.exists(depth_preview_path)
            else None
        ),
        "surface_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="surface_preview.jpg")
            if os.path.exists(surface_preview_path)
            else None
        ),
        "gravity_preview_url": (
            url_for("serve_frame", job_id=job_id, filename="gravity_preview.jpg")
            if os.path.exists(gravity_preview_path)
            else None
        ),
        "pest_mask_templates": pest_mask_templates,
        "frames": frames,
        "labels_json": json.dumps(labels_by_frame),
        "categories_json": json.dumps(categories_by_id),
        "render_width": render_width,
        "render_height": render_height,
        "playback_fps": _job_fps.get(job_id, 10),
        "render_time": _render_times.get(job_id),
    })


# ---------------------------------------------------------------------------
# Real Video Generator tab
# ---------------------------------------------------------------------------

@app.route("/real-generator", methods=["GET"])
def real_generator():
    batch_id = (request.args.get("batch_id") or "").strip()
    batch = _get_real_batch_snapshot(batch_id) if batch_id else None

    if batch_id and not batch:
        flash(f"Real generation batch not found: {batch_id}")

    default_form = {
        "length_seconds": 24,
        "fps": 10,
        "num_videos": 5,
        "generate_mp4": False,
    }
    context = {
        "real_form_values": default_form,
        "real_batch_state_path": GENERATED_STATE_PATH,
    }
    if batch:
        context.update({
            "real_batch_id": batch["batch_id"],
            "real_batch_done": batch["done"],
            "real_batch_progress_pct": batch["progress_pct"],
            "real_batch_requested": batch["requested"],
            "real_batch_completed": batch["completed"],
            "real_batch_generated": batch["generated"],
            "real_batch_workers": batch["workers"],
            "real_batch_time": batch["batch_time"],
            "real_batch_results": batch["results"],
            "real_batch_failures": batch["failures"],
            "real_batch_generate_mp4": batch["generate_mp4"],
            "real_batch_error": batch["error"],
            "real_form_values": batch["form_values"] or default_form,
        })

    return _render_real_generate_page(context, page_override=request.args.get("rpage", 1))


@app.route("/real-generator/upload", methods=["POST"])
def real_generator_upload():
    page = _parse_positive_int(request.form.get("rpage", 1), default=1)
    files = request.files.getlist("images")
    if not files:
        flash("No files selected for upload.")
        return redirect(url_for("real_generator", rpage=page))

    saved = 0
    skipped = 0
    for file in files:
        original = (file.filename or "").strip()
        if not original:
            skipped += 1
            continue
        if not _allowed_file(original):
            skipped += 1
            continue

        ext = f".{original.rsplit('.', 1)[1].lower()}"
        new_name = _allocate_kitchen_filename(ext)
        save_path = os.path.join(CURATED_IMG_DIR, new_name)
        file.save(save_path)
        saved += 1

    if saved:
        flash(f"Uploaded {saved} image(s) to curated_img/.")
    if skipped:
        flash(f"Skipped {skipped} file(s) due to invalid or missing extension.")
    return redirect(url_for("real_generator", rpage=page))


@app.route("/real-generator/generate", methods=["POST"])
def real_generator_generate():
    page = _parse_positive_int(request.form.get("rpage", 1), default=1)

    length_seconds = _parse_positive_float(request.form.get("length_seconds"), default=24.0)
    fps = _parse_positive_int(request.form.get("fps"), default=10)
    requested_videos = _parse_positive_int(request.form.get("num_videos"), default=1)
    generate_mp4 = str(request.form.get("generate_mp4", "")).strip().lower() in {"1", "true", "on", "yes"}

    curated = list_curated_images()
    if not curated:
        flash("No curated images available. Add images first.")
        return redirect(url_for("real_generator", rpage=page))

    pool = curated
    target_videos = requested_videos
    # Sampling with replacement lets us request more videos than unique kitchens.
    selected_filenames = random.choices(pool, k=target_videos)
    num_frames = max(1, int(round(length_seconds * fps)))
    try:
        split_map = _load_train_test_split()
    except (OSError, ValueError) as e:
        flash(f"Could not load train/test split: {e}")
        return redirect(url_for("real_generator", rpage=page))
    missing_kitchens = sorted(
        {name for name in selected_filenames if _kitchen_image_id(name) not in split_map}
    )
    if missing_kitchens:
        flash(
            "Missing train/test split assignment for: "
            + ", ".join(missing_kitchens[:5])
            + (" ..." if len(missing_kitchens) > 5 else "")
        )
        return redirect(url_for("real_generator", rpage=page))

    workers = _real_worker_count(target_videos)
    batch_id = uuid.uuid4().hex[:10]
    form_values = {
        "length_seconds": round(length_seconds, 2),
        "fps": fps,
        "num_videos": requested_videos,
        "generate_mp4": generate_mp4,
    }

    with _real_batches_lock:
        _real_batches[batch_id] = {
            "created_ts": time.time(),
            "done": False,
            "requested": target_videos,
            "completed": 0,
            "generated": 0,
            "workers": workers,
            "batch_time": None,
            "results": [],
            "failures": [],
            "error": None,
            "generate_mp4": generate_mp4,
            "form_values": form_values,
        }
        _prune_real_batches_locked()

    threading.Thread(
        target=_run_real_generation_batch,
        args=(
            batch_id,
            selected_filenames,
            split_map,
            length_seconds,
            fps,
            num_frames,
            workers,
            generate_mp4,
        ),
        daemon=True,
    ).start()

    flash(
        f"Started real generation batch {batch_id}: {target_videos} job(s) "
        f"with {workers} worker(s)."
    )
    return redirect(url_for("real_generator", rpage=page, batch_id=batch_id))


@app.route("/real-generator/status/<batch_id>", methods=["GET"])
def real_generator_status(batch_id):
    snapshot = _get_real_batch_snapshot(batch_id)
    if not snapshot:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify(snapshot)


# ---------------------------------------------------------------------------
# Kitchen Curator tab
# ---------------------------------------------------------------------------

@app.route("/curator", methods=["GET"])
def curator():
    try:
        idx = int(request.args.get("idx", 0))
    except ValueError:
        idx = 0

    images = list_local_kitchen_images()
    total = len(images)

    current = None
    depth_preview_url = None
    surface_preview_url = None
    analysis_error = None

    if total > 0:
        idx = max(0, min(idx, total - 1))
        current = images[idx]
        try:
            depth_name, surface_name = _ensure_curator_scene_analysis(current)
            if depth_name and surface_name:
                depth_preview_url = url_for("serve_kitchen_preview", filename=depth_name)
                surface_preview_url = url_for("serve_kitchen_preview", filename=surface_name)
        except Exception as e:
            analysis_error = f"Scene analysis failed for {current}: {e}"
    else:
        idx = 0

    with _download_status_lock:
        status = dict(_download_status)

    return render_template(
        "index.html",
        active_tab="curator",
        curator_images=images,
        curator_current=current,
        curator_idx=idx,
        curator_total=total,
        curator_depth_preview_url=depth_preview_url,
        curator_surface_preview_url=surface_preview_url,
        curator_analysis_error=analysis_error,
        curator_download_default=DEFAULT_DOWNLOAD_TARGET,
        download_running=status["running"],
        download_message=status["message"],
        download_status_json=json.dumps(status),
    )


@app.route("/curator/action", methods=["POST"])
def curator_action():
    action = (request.form.get("action") or "").strip().lower()
    filename = (request.form.get("filename") or "").strip()

    try:
        idx = int(request.form.get("idx", 0))
    except ValueError:
        idx = 0

    images = list_local_kitchen_images()
    if not filename or filename not in images:
        flash("Selected kitchen image was not found.")
        return redirect(url_for("curator", idx=idx))

    if action == "keep":
        src = os.path.join(UNCURATED_IMG_DIR, filename)
        ext = os.path.splitext(filename)[1].lower() or ".jpg"
        curated_name = _allocate_kitchen_filename(ext)
        dst = os.path.join(CURATED_IMG_DIR, curated_name)
        try:
            shutil.move(src, dst)
        except OSError as e:
            flash(f"Failed to move {filename} to curated_img/: {e}")
            return redirect(url_for("curator", idx=idx))
        # Also mark as seen so it isn't re-downloaded
        places_name = extract_places_filename(filename)
        if places_name:
            mark_places_as_seen({places_name})
            link_kitchen_to_places(curated_name, places_name)
        flash(f"Kept: {filename} as {curated_name} (moved to curated_img/)")
        remaining = list_local_kitchen_images()
        if not remaining:
            return redirect(url_for("curator"))
        next_idx = min(idx, len(remaining) - 1)
        return redirect(url_for("curator", idx=next_idx))

    if action == "delete":
        places_name = extract_places_filename(filename)
        if places_name:
            mark_places_as_seen({places_name})

        path = os.path.join(UNCURATED_IMG_DIR, filename)
        try:
            os.remove(path)
        except OSError as e:
            flash(f"Failed to delete {filename}: {e}")
            return redirect(url_for("curator", idx=idx))

        flash(f"Deleted {filename}")
        remaining = list_local_kitchen_images()
        if not remaining:
            return redirect(url_for("curator"))
        next_idx = min(idx, len(remaining) - 1)
        return redirect(url_for("curator", idx=next_idx))

    flash("Unknown action.")
    return redirect(url_for("curator", idx=idx))


@app.route("/download-more", methods=["POST"])
def download_more():
    """Trigger a background download of more kitchen images from Places365."""
    with _download_status_lock:
        if _download_status["running"]:
            flash("A download is already in progress.")
            return redirect(url_for("curator"))

    download_count = _parse_positive_int(
        request.form.get("download_count"),
        default=DEFAULT_DOWNLOAD_TARGET,
    )

    def _set_download_status(**updates):
        with _download_status_lock:
            _download_status.update(updates)

    def _do_download():
        _set_download_status(
            running=True,
            phase="starting",
            saved=0,
            total=0,
            current_file=None,
            message=f"Starting {DOWNLOAD_SPLIT} download (up to {download_count})...",
        )

        def _progress_cb(info):
            phase = info.get("phase")
            if phase == "scan":
                msg = (
                    f"Places365 kitchen pool={info.get('total_pool', 0)}, "
                    f"seen={info.get('seen', 0)}, unseen={info.get('unseen', 0)}."
                )
                _set_download_status(phase="scan", message=msg)
                return
            if phase == "downloading":
                saved = int(info.get("saved", 0))
                total = int(info.get("total", 0))
                current = info.get("current_file")
                msg = f"Downloading {saved}/{total}..."
                if current:
                    msg += f" {current}"
                _set_download_status(
                    phase="downloading",
                    saved=saved,
                    total=total,
                    current_file=current,
                    message=msg,
                )
                return
            if phase == "streaming":
                saved = int(info.get("saved", 0))
                total = int(info.get("total", 0))
                msg = info.get("message") or (
                    f"Streaming archive... {saved}/{total} downloaded so far."
                )
                _set_download_status(
                    phase="streaming",
                    saved=saved,
                    total=total,
                    current_file=None,
                    message=msg,
                )
                return

        try:
            from generator.kitchen_img.download_kitchens import main as _dl_main
            result = _dl_main(target=download_count, progress_cb=_progress_cb) or {}
            status = result.get("status")
            if status == "no_unseen":
                _set_download_status(
                    running=False,
                    phase="done",
                    saved=0,
                    total=0,
                    current_file=None,
                    message=(
                        "No unseen kitchen images left to download "
                        f"in {DOWNLOAD_SPLIT} (seen {result.get('seen', 0)} "
                        f"of {result.get('total_pool', 0)})."
                    ),
                )
            else:
                _set_download_status(
                    running=False,
                    phase="done",
                    saved=int(result.get("saved", 0)),
                    total=int(result.get("selected", 0)),
                    current_file=None,
                    message=(
                        f"Download complete: {result.get('saved', 0)} image(s) saved "
                        f"to uncurated_img/ ({DOWNLOAD_SPLIT})."
                    ),
                )
        except Exception as e:
            _set_download_status(
                running=False,
                phase="error",
                current_file=None,
                message=f"Download failed: {e}",
            )

    threading.Thread(target=_do_download, daemon=True).start()
    flash(
        f"Download started in the background from {DOWNLOAD_SPLIT} "
        f"(up to {download_count} images). "
        "This page will update automatically."
    )
    return redirect(url_for("curator"))


@app.route("/download-status", methods=["GET"])
def download_status():
    with _download_status_lock:
        status = dict(_download_status)
    status["uncurated_count"] = len(list_local_kitchen_images())
    return jsonify(status)


# ---------------------------------------------------------------------------
# Kitchen Image Generator tab (Gemini API)
# ---------------------------------------------------------------------------

def _gemini_key_status():
    """Return display info about the current GEMINI_API_KEY."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {"set": False, "masked": None}
    masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
    return {"set": True, "masked": masked}


@app.route("/kitchen-generator", methods=["GET"])
def kitchen_generator():
    from generator.kitchen_img.generate_kitchen import PROMPT_TEMPLATES
    return render_template(
        "index.html",
        active_tab="kitchen_generator",
        prompt_templates=PROMPT_TEMPLATES,
        gemini_key=_gemini_key_status(),
    )


@app.route("/kitchen-generator/set-api-key", methods=["POST"])
def kitchen_generator_set_api_key():
    """Save GEMINI_API_KEY to .env and update the running process."""
    new_key = (request.form.get("api_key") or "").strip()
    if not new_key:
        flash("API key cannot be empty.")
        return redirect(url_for("kitchen_generator"))

    # Validate rough format (Gemini keys start with "AIza")
    if not new_key.startswith("AIza"):
        flash("That doesn't look like a valid Gemini API key (should start with 'AIza').")
        return redirect(url_for("kitchen_generator"))

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        from dotenv import set_key
        set_key(env_path, "GEMINI_API_KEY", new_key)
    except Exception as e:
        flash(f"Failed to write .env: {e}")
        return redirect(url_for("kitchen_generator"))

    # Apply immediately to the running process
    os.environ["GEMINI_API_KEY"] = new_key
    flash("Gemini API key saved and active.")
    return redirect(url_for("kitchen_generator"))


@app.route("/kitchen-generator/generate", methods=["POST"])
def kitchen_generator_generate():
    """Call Gemini API to generate a kitchen image. Returns JSON."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set."}), 400

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        from generator.kitchen_img.generate_kitchen import generate_kitchen_image
        result = generate_kitchen_image(prompt, api_key)
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

    # Save generated image to uncurated_img/ so it can be curated like downloads.
    ext = "jpg" if "jpeg" in result["mime_type"] else "png"
    temp_name = f"gen_{uuid.uuid4().hex[:12]}.{ext}"
    temp_path = os.path.join(UNCURATED_IMG_DIR, temp_name)
    with open(temp_path, "wb") as f:
        f.write(result["image_bytes"])

    # Run depth/surface analysis
    depth_url = None
    surface_url = None
    try:
        depth_name, surface_name = _ensure_scene_analysis_for_path(temp_path)
        if depth_name:
            depth_url = url_for("serve_kitchen_preview", filename=depth_name)
        if surface_name:
            surface_url = url_for("serve_kitchen_preview", filename=surface_name)
    except Exception as e:
        print(f"Analysis failed for generated image: {e}")

    image_url = url_for("serve_kitchen_image", filename=temp_name)
    return jsonify({
        "image_url": image_url,
        "depth_url": depth_url,
        "surface_url": surface_url,
        "temp_filename": temp_name,
    })


@app.route("/kitchen-generator/save", methods=["POST"])
def kitchen_generator_save():
    """Save an approved generated image to curated_img/."""
    temp_filename = (request.form.get("temp_filename") or "").strip()
    if not temp_filename or not re.match(r"^gen_[a-f0-9]+\.(jpg|png)$", temp_filename):
        flash("Invalid temp filename.")
        return redirect(url_for("kitchen_generator"))

    src = os.path.join(UNCURATED_IMG_DIR, temp_filename)
    if not os.path.exists(src):
        flash("Generated image not found (may have expired). Please generate again.")
        return redirect(url_for("kitchen_generator"))

    ext = os.path.splitext(temp_filename)[1].lower() or ".jpg"
    curated_name = _allocate_kitchen_filename(ext)
    dst = os.path.join(CURATED_IMG_DIR, curated_name)
    try:
        shutil.move(src, dst)
    except OSError as e:
        flash(f"Failed to save image: {e}")
        return redirect(url_for("kitchen_generator"))

    flash(f"Image saved to curated_img/{curated_name}")
    return redirect(url_for("kitchen_generator"))


@app.route("/kitchen-generator/discard", methods=["POST"])
def kitchen_generator_discard():
    """Delete a temp generated image."""
    temp_filename = (request.form.get("temp_filename") or "").strip()
    if temp_filename and re.match(r"^gen_[a-f0-9]+\.(jpg|png)$", temp_filename):
        path = os.path.join(UNCURATED_IMG_DIR, temp_filename)
        try:
            os.remove(path)
        except OSError:
            pass
    flash("Image discarded.")
    return redirect(url_for("kitchen_generator"))


# ---------------------------------------------------------------------------
# Misc routes
# ---------------------------------------------------------------------------

@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate(job_id):
    image_path = _source_images.get(job_id)
    page = _parse_positive_int(request.form.get("page", 1), default=1)
    length_seconds = _parse_positive_float(request.form.get("length_seconds"), default=24.0)
    fps = _parse_positive_int(request.form.get("fps"), default=10)
    num_frames = max(1, int(round(length_seconds * fps)))
    if not image_path or not os.path.exists(image_path):
        flash("Source image no longer available — please upload again.")
        return redirect(url_for("index", page=page, length_seconds=length_seconds, fps=fps))

    try:
        result = _generate_video_for_image_with_params(
            image_path,
            num_frames=num_frames,
            fps=fps,
            use_real_outputs=False,
            assemble_video=True,
        )
    except Exception as e:
        flash(f"Regeneration failed: {e}")
        return redirect(url_for("results", job_id=job_id, page=page, length_seconds=length_seconds, fps=fps))

    return redirect(url_for(
        "results",
        job_id=(result.get("video_id") or result.get("job_id")),
        page=page,
        length_seconds=length_seconds,
        fps=fps,
    ))


@app.route("/outputs/videos/<job_id>.mp4")
def serve_video(job_id):
    return send_from_directory(VIDEOS_DIR, f"{job_id}.mp4")


@app.route("/outputs/frames/<job_id>/<filename>")
def serve_frame(job_id, filename):
    return send_from_directory(os.path.join(FRAMES_DIR, job_id), filename)


@app.route("/kitchen-images/<path:filename>")
def serve_kitchen_image(filename):
    """Serve uncurated kitchen images (downloaded/generated, pending review)."""
    return send_from_directory(UNCURATED_IMG_DIR, filename)


@app.route("/curated-images/<path:filename>")
def serve_curated_image(filename):
    """Serve curated (approved) kitchen images."""
    return send_from_directory(CURATED_IMG_DIR, filename)


@app.route("/kitchen-previews/<path:filename>")
def serve_kitchen_preview(filename):
    return send_from_directory(CURATOR_PREVIEW_DIR, filename)


@app.route("/outputs/uploads/<filename>")
def serve_upload(filename):
    """Serve temp uploaded/generated images."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/cleanup")
def cleanup():
    """Delete all generated outputs (testing convenience)."""
    for d in [FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, UPLOAD_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
    flash("All outputs cleared.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        if TESTING_MODE:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
            print(f"Cleaned up temp dir: {_TEMP_DIR}")
