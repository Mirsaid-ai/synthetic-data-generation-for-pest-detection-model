"""
Upscale kitchen images from 256×256 → 1024×1024 using Real-ESRGAN (4× model).

Usage:
    python scripts/upscale_images.py
    python scripts/upscale_images.py --input_dir generator/kitchen_img/curated_img --scale 4

Overwrites input files in-place by default. Use --output_dir to write elsewhere.
"""

import argparse
import sys
import os
from pathlib import Path

def install_deps():
    """Check and install Real-ESRGAN if missing."""
    try:
        import realesrgan  # noqa: F401
    except ImportError:
        print("Real-ESRGAN not found. Installing...")
        os.system(f"{sys.executable} -m pip install realesrgan basicsr facexlib gfpgan")
        print("Installation complete.\n")


def _patch_torchvision_compat():
    """basicsr<1.4.3 imports torchvision.transforms.functional_tensor which was
    removed in torchvision>=0.17. Shim the missing module before basicsr loads."""
    import sys
    import types
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional as _F
            _shim = types.ModuleType("torchvision.transforms.functional_tensor")
            _shim.rgb_to_grayscale = _F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = _shim
        except Exception:
            pass


def build_upsampler(scale: int = 4, half: bool = False):
    _patch_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if scale == 4:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        model_name = "RealESRGAN_x4plus"
    elif scale == 2:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=2
        )
        model_name = "RealESRGAN_x2plus"
    else:
        raise ValueError(f"Unsupported scale {scale}. Use 2 or 4.")

    # RealESRGANer auto-downloads weights to ~/.cache/realesrgan/
    weights_url = (
        f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=weights_url,
        model=model,
        tile=256,        # tile size — keeps memory low for large batches
        tile_pad=10,
        pre_pad=0,
        half=half,       # fp16 on GPU; keep False on MPS/CPU for stability
    )
    return upsampler


def upscale_directory(
    input_dir: Path,
    output_dir: Path | None,
    scale: int = 4,
    exts: tuple = (".jpg", ".jpeg", ".png"),
    half: bool = False,
    dry_run: bool = False,
):
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        print(f"No images found in {input_dir}")
        return

    in_place = output_dir is None
    if not in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} image(s) in {input_dir}")
    print(f"Scale: {scale}×  |  Output: {'in-place' if in_place else output_dir}\n")

    if dry_run:
        for p in images:
            print(f"  [dry-run] would upscale: {p.name}")
        return

    install_deps()
    import cv2  # noqa: PLC0415 — imported after install_deps ensures it exists
    upsampler = build_upsampler(scale=scale, half=half)

    ok, failed = 0, []
    for i, img_path in enumerate(images, 1):
        dest = img_path if in_place else output_dir / img_path.name
        print(f"[{i:3d}/{len(images)}] {img_path.name} ", end="", flush=True)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print("→ SKIP (could not read)")
            failed.append(img_path.name)
            continue

        h, w = img_bgr.shape[:2]
        try:
            out_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
            oh, ow = out_bgr.shape[:2]

            if dest.suffix.lower() in (".jpg", ".jpeg"):
                cv2.imwrite(str(dest), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(dest), out_bgr)

            print(f"→ {w}×{h}  →  {ow}×{oh}  ✓")
            ok += 1
        except Exception as e:
            print(f"→ ERROR: {e}")
            failed.append(img_path.name)

    print(f"\nDone. {ok}/{len(images)} upscaled successfully.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(description="Upscale kitchen images with Real-ESRGAN")
    parser.add_argument(
        "--input_dir",
        default="generator/kitchen_img/curated_img",
        help="Directory of input images (default: generator/kitchen_img/curated_img)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Omit to overwrite input files in-place.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="Upscale factor (default: 4)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use fp16 (faster on CUDA GPU, may be unstable on MPS/CPU)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List files that would be processed without doing anything",
    )
    args = parser.parse_args()

    # Resolve relative paths from repo root
    repo_root = Path(__file__).parent.parent
    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve() if args.output_dir else None

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    upscale_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        scale=args.scale,
        half=args.half,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
