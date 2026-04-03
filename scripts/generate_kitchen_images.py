"""
Automated kitchen image generation using the Gemini API.

Generates photorealistic kitchen images from a curated prompt list and saves
them directly to the output directory. Supports both batch (automated) and
interactive (manual review) modes.

Usage — fully automated, save directly to curated_img:
    python scripts/generate_kitchen_images.py --count 3

Usage — save to a staging folder for manual review first:
    python scripts/generate_kitchen_images.py --count 3 --staging

Usage — single custom prompt:
    python scripts/generate_kitchen_images.py --prompt "Your prompt here" --count 5

Usage — also mirror saves to the web app's approved_images (keeps the gallery usable):
    python scripts/generate_kitchen_images.py --count 3 --mirror_web_app

API key: Set GEMINI_API_KEY in environment or in kitchen_image_gen/.env.local
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Prompt library ─────────────────────────────────────────────────────────────

PROMPT_PREFIX = (
    "Generate a photorealistic image of a commercial kitchen interior. "
    "No people or animals should be present. "
)

# Each entry: (prompt_suffix, weight)
# weight controls how many times the prompt is sampled relative to others.
PROMPT_LIBRARY = [
    # ── Overhead / CCTV angle (highest priority — matches test environment) ────
    (
        "Commercial cafeteria kitchen floor, overhead CCTV camera angle, "
        "ceramic tile floor, fluorescent lighting, photorealistic, empty",
        3,
    ),
    (
        "University cafeteria kitchen, bird's eye view from ceiling security camera, "
        "stainless steel equipment, white tile floor, bright overhead lights, empty floor",
        3,
    ),
    (
        "Restaurant kitchen floor from ceiling-mounted security camera, wide angle, "
        "industrial kitchen, no pests, clean tile floor",
        3,
    ),
    (
        "Institutional kitchen, top-down view, commercial cooking equipment, "
        "sealed concrete floor, overhead fluorescent lighting",
        2,
    ),
    (
        "School cafeteria kitchen viewed from ceiling camera, beige vinyl floor, "
        "institutional counters, ceiling grid lights, wide angle overhead shot",
        2,
    ),
    # ── Counter / table surfaces (pests can be here too) ─────────────────────
    (
        "Commercial kitchen countertop from overhead camera angle, stainless steel prep "
        "surface, kitchen equipment visible, top-down view, no people",
        2,
    ),
    (
        "Restaurant kitchen prep counter, top-down bird's eye view, cutting boards, "
        "utensils, stainless steel surface, bright overhead lights",
        2,
    ),
    (
        "Cafeteria kitchen serving counter viewed from ceiling security camera, "
        "tray rails, food trays, institutional lighting, wide angle",
        2,
    ),
    # ── Diverse angles / lighting (adds domain diversity) ────────────────────
    (
        "Large commercial kitchen with white tile floor, stainless steel counters, "
        "overhead fluorescent lighting, wide angle shot from corner",
        1,
    ),
    (
        "Restaurant kitchen with dark stone floor, industrial shelving, "
        "warm tungsten lighting, shot from doorway perspective",
        1,
    ),
    (
        "Small cafe kitchen with checkered linoleum floor, wooden counters, "
        "natural window light, slightly elevated camera angle",
        1,
    ),
    (
        "Industrial kitchen with concrete floor, metal prep tables, "
        "harsh overhead LED panels, straight-on view",
        1,
    ),
    (
        "Hotel kitchen with gray tile floor, marble counters, "
        "mixed warm and cool lighting, diagonal perspective",
        1,
    ),
    (
        "Home kitchen floor, side angle from counter height, "
        "linoleum floor, natural window light",
        1,
    ),
    (
        "Fast food kitchen with red tile floor, stainless steel equipment, "
        "bright fluorescent lights, low angle shot",
        1,
    ),
    (
        "Night-mode commercial kitchen under IR-style monochrome lighting, "
        "grayscale tones, security camera aesthetic, empty floor, overhead view",
        1,
    ),
]


def build_weighted_prompt_list() -> list[str]:
    """Expand PROMPT_LIBRARY by weight into a flat list."""
    prompts = []
    for suffix, weight in PROMPT_LIBRARY:
        prompts.extend([suffix] * weight)
    return prompts


# ── API key loading ────────────────────────────────────────────────────────────

def load_api_key(repo_root: Path) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        env_file = repo_root / "kitchen_image_gen" / ".env.local"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print(
            "Error: GEMINI_API_KEY not found.\n"
            "Set it as an environment variable or add it to kitchen_image_gen/.env.local:\n"
            "  GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)
    return key


# ── Gemini client ──────────────────────────────────────────────────────────────

def get_client(api_key: str):
    try:
        from google import genai  # noqa: PLC0415
    except ImportError:
        print("Installing google-genai...")
        os.system(f"{sys.executable} -m pip install google-genai -q")
        from google import genai  # noqa: PLC0415
    return genai.Client(api_key=api_key)


MODEL = "gemini-3.1-flash-image-preview"


def generate_image(client, prompt_suffix: str) -> tuple[bytes, str] | None:
    """Returns (image_bytes, mime_type) or None on failure."""
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415

    full_prompt = PROMPT_PREFIX + prompt_suffix
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                data = part.inline_data.data
                mime = part.inline_data.mime_type or "image/png"
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return data, mime
        print("  → No image in response (model returned text only)")
        return None
    except Exception as e:
        print(f"  → API error: {e}")
        return None


# ── File saving ────────────────────────────────────────────────────────────────

def mime_to_ext(mime: str) -> str:
    return {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")


def next_filename(output_dir: Path, ext: str) -> Path:
    """Find the next available kitchen_XXXX filename."""
    existing = sorted(output_dir.glob("kitchen_*" + ext))
    if not existing:
        return output_dir / f"kitchen_gen_0001{ext}"
    last = existing[-1].stem  # e.g. kitchen_gen_0023
    parts = last.rsplit("_", 1)
    try:
        n = int(parts[-1]) + 1
    except ValueError:
        n = len(existing) + 1
    return output_dir / f"kitchen_gen_{n:04d}{ext}"


def save_image(
    image_bytes: bytes,
    mime: str,
    prompt: str,
    output_dir: Path,
    web_app_dir: Path | None = None,
) -> Path:
    ext = mime_to_ext(mime)
    dest = next_filename(output_dir, ext)
    dest.write_bytes(image_bytes)

    # Save companion metadata JSON
    meta = {
        "prompt": PROMPT_PREFIX + prompt,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": dest.name,
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    # Optionally mirror to the web app's approved_images gallery
    if web_app_dir:
        web_app_dir.mkdir(parents=True, exist_ok=True)
        import uuid
        uid = uuid.uuid4().hex[:8]
        web_dest = web_app_dir / f"kitchen_{uid}{ext}"
        web_dest.write_bytes(image_bytes)
        web_meta = {
            "id": uid,
            "prompt": PROMPT_PREFIX + prompt,
            "filename": web_dest.name,
            "timestamp": meta["timestamp"],
        }
        (web_app_dir / f"kitchen_{uid}.json").write_text(json.dumps(web_meta, indent=2))

    return dest


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate kitchen images via Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of images to generate (default: 10)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Use a single custom prompt instead of the built-in library",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save images. Default: generator/kitchen_img/curated_img",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Save to generator/kitchen_img/generated_img/ for manual review instead",
    )
    parser.add_argument(
        "--mirror_web_app",
        action="store_true",
        help="Also copy each image to kitchen_image_gen/public/approved_images/",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2.0)",
    )
    parser.add_argument(
        "--list_prompts",
        action="store_true",
        help="Print the full prompt library and exit",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    if args.list_prompts:
        print(f"Prompt prefix: {PROMPT_PREFIX!r}\n")
        for i, (suffix, weight) in enumerate(PROMPT_LIBRARY, 1):
            print(f"[{i:2d}] weight={weight}  {suffix}")
        return

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.staging:
        output_dir = repo_root / "generator" / "kitchen_img" / "generated_img"
        print(f"Staging mode: images will be saved to {output_dir}")
        print("Review them and move the keepers to generator/kitchen_img/curated_img/\n")
    else:
        output_dir = repo_root / "generator" / "kitchen_img" / "curated_img"
    output_dir.mkdir(parents=True, exist_ok=True)

    web_app_dir = (
        repo_root / "kitchen_image_gen" / "public" / "approved_images"
        if args.mirror_web_app
        else None
    )

    api_key = load_api_key(repo_root)
    client = get_client(api_key)

    # Build prompt sequence
    if args.prompt:
        prompts = [args.prompt] * args.count
    else:
        pool = build_weighted_prompt_list()
        import random
        random.shuffle(pool)
        # Cycle through pool to reach requested count
        prompts = [pool[i % len(pool)] for i in range(args.count)]

    print(f"Generating {args.count} image(s) → {output_dir}\n")

    ok, failed = 0, 0
    for i, prompt in enumerate(prompts, 1):
        short = prompt[:70] + "..." if len(prompt) > 70 else prompt
        print(f"[{i:3d}/{args.count}] {short}")

        result = generate_image(client, prompt)
        if result is None:
            failed += 1
        else:
            image_bytes, mime = result
            dest = save_image(image_bytes, mime, prompt, output_dir, web_app_dir)
            print(f"          → saved: {dest.name}  ({len(image_bytes)//1024} KB)")
            ok += 1

        if i < args.count:
            time.sleep(args.delay)

    print(f"\nDone. {ok} saved, {failed} failed.")
    if args.staging:
        print(f"\nReview images in:\n  {output_dir}")
        print("Then move keepers to:\n  generator/kitchen_img/curated_img/")


if __name__ == "__main__":
    main()
