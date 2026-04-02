"""Download kitchen images from Places365 Standard (train/val split).

Strategy:
  1. Stream filelist tar → extract the configured split file in memory
  2. Filter for category 203 (kitchen) to get target filenames
  3. Skip filenames we have seen before (downloaded or curator-deleted)
  4. Stream the configured image archive → save only unseen kitchen images,
     stop at TARGET

No API key required. All data is MIT/CSAIL public.

Images are saved to generator/kitchen_img/uncurated_img/ as a staging
area. Curated images are moved to curated_img/ by the web app.
"""

import io
import json
import os
import re
import tarfile
import urllib.request
from datetime import datetime

# Images are saved to generator/kitchen_img/uncurated_img/
KITCHEN_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(KITCHEN_ROOT_DIR, "uncurated_img")
TARGET = 100
KITCHEN_CATEGORY = 203   # /k/kitchen in Places365-standard
STATE_FILE = os.path.join(KITCHEN_ROOT_DIR, "download_state.json")
DATASET_SPLIT = "train_standard"  # options: "train_standard", "val"

FILELIST_URL = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
SPLIT_TO_MEMBER = {
    "train_standard": "places365_train_standard.txt",
    "val": "places365_val.txt",
}
SPLIT_TO_IMAGES_URL = {
    "train_standard": "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar",
    "val": "https://data.csail.mit.edu/places/places365/val_256.tar",
}
TRAIN_ARCHIVE_PREFIX = "data_256/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PLACES_NAME_RE = re.compile(r"(Places365_val_\d{8}\.jpg)$")
OUTPUT_INDEX_RE = re.compile(r"^kitchen_(\d+)_")
OUTPUT_SOURCE_RE = re.compile(r"^kitchen_\d+_(.+)$")

def fetch_bytes(url, label):
    print(f"  Downloading {label} ...", flush=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _normalize_source_id(name):
    return (name or "").strip().lstrip("/")


def _source_id_to_token(source_id):
    """Encode source id into a filename-safe token."""
    return _normalize_source_id(source_id).replace("/", "__")


def _token_to_source_id(token):
    """Decode filename token back to source id."""
    return (token or "").replace("__", "/")


def extract_places_filename(name):
    """Extract canonical source id from local output filename.

    Returns:
        - Val split: Places365_val_XXXXXXXX.jpg
        - Train split: k/kitchen/XXXXXXXX.jpg
        - None if pattern not recognized.
    """
    m = PLACES_NAME_RE.search(name)
    if m:
        return m.group(1)

    base = os.path.basename(name)
    m2 = OUTPUT_SOURCE_RE.match(base)
    if not m2:
        return None
    token = m2.group(1)
    if "__" not in token:
        return None
    decoded = _token_to_source_id(token)
    if decoded.startswith("k/kitchen/") and decoded.lower().endswith(".jpg"):
        return decoded
    return None


def list_local_kitchen_images():
    """Return local kitchen image files in OUT_DIR (staging area)."""
    files = []
    for name in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            files.append(name)
    return files


def _empty_state():
    return {
        "seen_places365_files": [],
        "kitchen_mappings": {},
    }


def _places365_link_for_source(source_id):
    source_id = _normalize_source_id(source_id)
    if not source_id:
        return None
    if source_id.startswith("Places365_val_"):
        return f"https://data.csail.mit.edu/places/places365/val_256/{source_id}"
    if source_id.startswith("k/kitchen/"):
        return (
            "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
            f"#{source_id}"
        )
    return None


def _load_state():
    if not os.path.exists(STATE_FILE):
        return _empty_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_state()

    if not isinstance(data, dict):
        return _empty_state()

    state = _empty_state()
    seen = data.get("seen_places365_files", [])
    if isinstance(seen, list):
        state["seen_places365_files"] = sorted({v for v in seen if isinstance(v, str) and v})

    mappings = {}
    for key in ("kitchen_mappings", "kitchen_img_mappings"):
        candidate = data.get(key, {})
        if isinstance(candidate, dict):
            mappings.update(candidate)
    if isinstance(mappings, dict):
        clean = {}
        for kitchen_id, meta in mappings.items():
            kid = _normalize_kitchen_id(kitchen_id)
            if not kid:
                continue
            if isinstance(meta, dict):
                clean[kid] = {
                    "places365_source_id": meta.get("places365_source_id"),
                    "places365_link": meta.get("places365_link"),
                    "linked_at": meta.get("linked_at"),
                }
        state["kitchen_mappings"] = clean
    return state


def _save_state(payload):
    final_payload = _empty_state()
    final_payload["seen_places365_files"] = sorted(
        {v for v in payload.get("seen_places365_files", []) if isinstance(v, str) and v}
    )
    mappings = payload.get("kitchen_mappings", {})
    if isinstance(mappings, dict):
        final_payload["kitchen_mappings"] = mappings
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)


def _load_seen_from_state():
    data = _load_state()
    return set(data.get("seen_places365_files", []))


def _seen_from_existing_files():
    seen = set()
    for name in list_local_kitchen_images():
        places_name = extract_places_filename(name)
        if places_name:
            seen.add(places_name)
    return seen


def get_seen_places():
    """Union of state file + current images on disk."""
    return _load_seen_from_state() | _seen_from_existing_files()


def mark_places_as_seen(places_filenames):
    """Persist given Places filenames so they are never re-downloaded."""
    clean = {p for p in places_filenames if isinstance(p, str) and p}
    if not clean:
        return
    seen = get_seen_places()
    seen.update(clean)
    state = _load_state()
    state["seen_places365_files"] = sorted(seen)
    _save_state(state)


def _normalize_kitchen_id(kitchen_id):
    kitchen_id = (kitchen_id or "").strip()
    if not kitchen_id:
        return None
    if kitchen_id.startswith("kitchen_img_"):
        kitchen_id = "kitchen_" + kitchen_id[len("kitchen_img_"):]
    return kitchen_id


def link_kitchen_to_places(kitchen_id, places_source_id):
    kitchen_id = _normalize_kitchen_id(kitchen_id)
    places_source_id = _normalize_source_id(places_source_id)
    if not kitchen_id or not places_source_id:
        return

    state = _load_state()
    mappings = state.get("kitchen_mappings") or {}
    mappings[kitchen_id] = {
        "places365_source_id": places_source_id,
        "places365_link": _places365_link_for_source(places_source_id),
        "linked_at": datetime.now().isoformat(timespec="seconds"),
    }
    state["kitchen_mappings"] = mappings
    _save_state(state)


def link_kitchen_img_to_places(kitchen_img_id, places_source_id):
    """Backward-compatible alias."""
    link_kitchen_to_places(kitchen_img_id, places_source_id)


def next_output_index():
    """Return next sequential output index for kitchen_<idx>_*.jpg filenames."""
    max_idx = 0
    for name in os.listdir(OUT_DIR):
        m = OUTPUT_INDEX_RE.match(name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _member_source_id(member_name, split):
    """Map a tar member path to the canonical source id used by filelists/state."""
    member_name = (member_name or "").strip()
    if split == "val":
        return os.path.basename(member_name)
    if split == "train_standard":
        if member_name.startswith(TRAIN_ARCHIVE_PREFIX):
            member_name = member_name[len(TRAIN_ARCHIVE_PREFIX):]
        return _normalize_source_id(member_name)
    return _normalize_source_id(member_name)


def get_kitchen_filenames(split=DATASET_SPLIT):
    """Return set of source ids for category-203 kitchen images in chosen split."""
    member_name = SPLIT_TO_MEMBER.get(split)
    if not member_name:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {sorted(SPLIT_TO_MEMBER)}")

    data = fetch_bytes(FILELIST_URL, "filelist tar (67 MB)")
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        file_member = t.getmember(member_name)
        file_txt = t.extractfile(file_member).read().decode("utf-8")

    kitchen = set()
    for line in file_txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and int(parts[1]) == KITCHEN_CATEGORY:
            kitchen.add(_normalize_source_id(parts[0]))

    print(f"  {len(kitchen)} kitchen images in split '{split}'.")
    return kitchen


def stream_download_kitchen(kitchen_files, start_index, split=DATASET_SPLIT, progress_cb=None):
    """Stream Places365 split archive and save only selected kitchen images."""
    images_url = SPLIT_TO_IMAGES_URL.get(split)
    if not images_url:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {sorted(SPLIT_TO_IMAGES_URL)}")

    print(
        f"  Streaming split archive ({split}) — saving up to {len(kitchen_files)} images ...",
        flush=True,
    )
    if progress_cb:
        progress_cb({
            "phase": "streaming",
            "saved": 0,
            "total": len(kitchen_files),
            "split": split,
            "message": (
                f"Streaming Places365 {split} archive. "
                "First save may take time for train_standard."
            ),
        })
    req = urllib.request.Request(images_url, headers=HEADERS)

    saved = 0
    saved_places = set()
    out_index = start_index
    with urllib.request.urlopen(req, timeout=300) as r:
        with tarfile.open(fileobj=r, mode="r|") as tar:
            for member in tar:
                source_id = _member_source_id(member.name, split)
                if not member.isfile() or source_id not in kitchen_files:
                    continue

                fobj = tar.extractfile(member)
                if fobj is None:
                    continue

                token = _source_id_to_token(source_id)
                out_path = os.path.join(OUT_DIR, f"kitchen_{out_index:03d}_{token}")
                with open(out_path, "wb") as f:
                    f.write(fobj.read())
                saved += 1
                out_index += 1
                saved_places.add(source_id)
                print(f"  [{saved}/{len(kitchen_files)}] {source_id}", flush=True)
                if progress_cb:
                    progress_cb({
                        "phase": "downloading",
                        "saved": saved,
                        "total": len(kitchen_files),
                        "current_file": source_id,
                    })

                if saved >= len(kitchen_files):
                    break

    print(f"\nDone — {saved} images saved to:\n  {OUT_DIR}")
    return saved_places


def main(target=TARGET, progress_cb=None, split=DATASET_SPLIT):
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        requested = int(target)
    except (TypeError, ValueError):
        requested = TARGET
    if requested <= 0:
        requested = TARGET

    kitchen_files = get_kitchen_filenames(split=split)
    seen_places = get_seen_places()
    unseen_kitchen_files = sorted(kitchen_files - seen_places)
    print(f"  Already seen (downloaded or curated): {len(seen_places)}")
    if progress_cb:
        progress_cb({
            "phase": "scan",
            "requested": requested,
            "split": split,
            "seen": len(seen_places),
            "total_pool": len(kitchen_files),
            "unseen": len(unseen_kitchen_files),
        })

    if not unseen_kitchen_files:
        print("  No unseen kitchen images left to download.")
        return {
            "status": "no_unseen",
            "requested": requested,
            "saved": 0,
            "selected": 0,
            "seen": len(seen_places),
            "unseen": 0,
            "total_pool": len(kitchen_files),
        }

    target_count = min(requested, len(unseen_kitchen_files))
    # Deterministic sequential selection: pick the next unseen files in order.
    selected_files = set(unseen_kitchen_files[:target_count])
    print(f"  Selecting next {target_count} unseen images (requested={requested}, split={split}).")

    start_index = next_output_index()
    saved_places = stream_download_kitchen(
        selected_files,
        start_index=start_index,
        split=split,
        progress_cb=progress_cb,
    )
    mark_places_as_seen(saved_places)
    return {
        "status": "downloaded",
        "requested": requested,
        "saved": len(saved_places),
        "selected": target_count,
        "seen": len(seen_places) + len(saved_places),
        "unseen": max(len(unseen_kitchen_files) - len(saved_places), 0),
        "total_pool": len(kitchen_files),
    }


if __name__ == "__main__":
    main()
