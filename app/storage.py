"""Storage shim for local-vs-cloud deployments.

Resolves paths for the model checkpoint and optional Supabase integration
behind env vars so the app runs identically on a laptop, an HF Space, or a
container with Supabase-backed storage.

Env vars (all optional):
    MODEL_PATH        Explicit local path to best.pt. Wins over everything.
    CHECKPOINT_URL    Download URL (HF Hub or signed Supabase URL) used once
                      to populate the resolved path when the file is missing.
    HF_DATA_DIR       HF Space persistent volume (default: /data).
    SUPABASE_URL      Supabase project URL (optional).
    SUPABASE_SERVICE_KEY   Service-role key (optional).
    SUPABASE_BUCKET   Bucket name holding shared artifacts (optional).

Notes
-----
- Nothing in this module is required for local dev. If all env vars are
  unset, `get_checkpoint_path()` returns ./checkpoints/best.pt.
- Network downloads only run when `ensure_checkpoint()` is called and the
  target path does not yet exist, so repeated calls are idempotent.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"


# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------

def _hf_data_dir() -> Path:
    return Path(os.environ.get("HF_DATA_DIR", "/data"))


def get_checkpoint_path() -> str:
    """Return the path we should try to load the detector from.

    Order:
      1. $MODEL_PATH                 (explicit override)
      2. $HF_DATA_DIR/best.pt        (HF persistent volume, typical cloud path)
      3. ./checkpoints/best.pt       (local default)
    """
    explicit = os.environ.get("MODEL_PATH", "").strip()
    if explicit:
        return str(Path(explicit).expanduser())

    hf_path = _hf_data_dir() / "best.pt"
    if hf_path.is_file():
        return str(hf_path)

    return str(DEFAULT_LOCAL_CHECKPOINT)


def ensure_checkpoint() -> Optional[str]:
    """Download the checkpoint once if `CHECKPOINT_URL` is set and the target
    path is empty. Returns the resolved path (whether or not a download
    happened), or None when no checkpoint is configured at all.

    Safe to call at app startup. Network failures are swallowed with a log
    message so the app still boots and the UI can display "No model loaded".
    """
    target = Path(get_checkpoint_path())
    if target.is_file():
        return str(target)

    url = os.environ.get("CHECKPOINT_URL", "").strip()
    if not url:
        # Best effort: create the parent dir so an admin can drop best.pt in
        # later via the HF file manager without manually creating /data.
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return str(target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[storage] Could not create checkpoint dir {target.parent}: {e}")
        return str(target)

    # Lazy imports: keep the app importable even when `requests` / HF hub
    # aren't available in the dev env.
    try:
        if "huggingface.co" in url:
            _download_from_hf(url, target)
        else:
            _download_via_http(url, target)
        print(f"[storage] Downloaded checkpoint to {target}")
    except Exception as e:
        print(f"[storage] Failed to download checkpoint from {url}: {e}")

    return str(target)


def _download_via_http(url: str, target: Path) -> None:
    import requests

    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    shutil.move(tmp, target)


def _download_from_hf(url: str, target: Path) -> None:
    """Download a file from a https://huggingface.co/... URL.

    Accepts both `blob/main/...` and `resolve/main/...` variants; normalizes
    to `resolve/main/...` because `blob` returns HTML.
    """
    if "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    _download_via_http(url, target)


# ---------------------------------------------------------------------------
# Supabase (optional)
# ---------------------------------------------------------------------------

_supabase_client_cache: Optional[Any] = None
_supabase_client_loaded = False


def ensure_supabase_client() -> Optional[Any]:
    """Return a supabase-py client when the env vars are set; else None.

    Callers must handle `None` by falling back to the local filesystem.
    This keeps Supabase a truly optional dependency for the grade.
    """
    global _supabase_client_cache, _supabase_client_loaded
    if _supabase_client_loaded:
        return _supabase_client_cache

    _supabase_client_loaded = True

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        return None

    try:
        from supabase import create_client
    except ImportError:
        print(
            "[storage] SUPABASE_URL/SERVICE_KEY set but supabase-py is not "
            "installed. pip install supabase to enable cloud storage."
        )
        return None

    try:
        _supabase_client_cache = create_client(url, key)
    except Exception as e:
        print(f"[storage] Failed to initialize Supabase client: {e}")
        _supabase_client_cache = None
    return _supabase_client_cache


def supabase_bucket_name() -> Optional[str]:
    name = os.environ.get("SUPABASE_BUCKET", "").strip()
    return name or None


# ---------------------------------------------------------------------------
# Diagnostics — handy for the /inference model-load widget and README_DEPLOY.
# ---------------------------------------------------------------------------

def storage_status() -> dict:
    target = get_checkpoint_path()
    return {
        "checkpoint_path": target,
        "checkpoint_exists": os.path.isfile(target),
        "checkpoint_url_set": bool(os.environ.get("CHECKPOINT_URL")),
        "hf_data_dir": str(_hf_data_dir()),
        "supabase_configured": bool(
            os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")
        ),
        "model_path_override": os.environ.get("MODEL_PATH", "") or None,
    }
