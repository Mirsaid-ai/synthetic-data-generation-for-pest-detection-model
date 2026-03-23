"""
generate_floor_mask.py
======================
Generates a binary floor mask from a kitchen image using semantic segmentation.

  White (255) = walkable floor
  Black (0)   = walls, cabinets, counters, ceiling

Uses SegFormer-B2 pretrained on ADE20K.

Install:
    pip install torch torchvision transformers pillow opencv-python numpy scipy

Usage:
    python generate_floor_mask.py --image kitchen1.png --output kitchen1_mask.png --debug

    # With depth map refinement:
    python generate_floor_mask.py --image kitchen1.png --depth kitchen1_depth.png \
                                  --output kitchen1_mask.png --debug

Optional flags:
    --model          HuggingFace model (default: nvidia/segformer-b2-finetuned-ade-512-512)
    --depth          Grayscale depth map to AND with floor result
    --depth_thresh   Depth cutoff 0-255; pixels above = floor (default: 40)
    --floor_labels   ADE20K label indices for floor (default: 3)
                     Add more if missed: --floor_labels 3 6 52
    --smooth_px      Boundary smooth radius px (default: 5, 0 to disable)
    --debug          Save colour-coded debug overlay PNG

ADE20K floor labels:
    3=floor  6=road  52=rug/carpet
"""

import cv2
import numpy as np
import argparse
import os
import sys


def check_imports():
    missing = []
    for pkg, pip in [("torch","torch"),("transformers","transformers"),
                     ("PIL","pillow"),("scipy","scipy")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip)
    if missing:
        print(f"[ERROR] Missing: pip install {' '.join(missing)}")
        sys.exit(1)


ADE20K_NAMES = {
    0:"wall", 1:"building", 2:"sky", 3:"floor", 4:"tree", 5:"ceiling",
    6:"road", 7:"bed", 8:"windowpane", 9:"grass", 10:"cabinet",
    11:"sidewalk", 14:"door", 15:"table", 19:"chair", 24:"shelf",
    26:"mirror", 32:"desk", 34:"wardrobe", 35:"lamp", 44:"counter",
    46:"sink", 49:"refrigerator", 52:"stairs", 58:"stairway",
    63:"coffee table", 64:"toilet", 69:"countertop", 70:"stove",
}


# ─────────────────────────────────────────────
#  SEGMENTATION
# ─────────────────────────────────────────────

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_segformer(image_bgr, model_name):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    from PIL import Image

    device = get_device()
    print(f"[INFO] Loading model: {model_name}  (device: {device})")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model     = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    model.to(device)

    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    print("[INFO] Running segmentation…")
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_bgr.shape[:2]
    up = torch.nn.functional.interpolate(
        outputs.logits.cpu(), size=(h, w), mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze().numpy().astype(np.int32)


def labels_to_mask(seg_map, labels):
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for lbl in labels:
        mask[seg_map == lbl] = 255
    return mask


# ─────────────────────────────────────────────
#  POST-PROCESSING
#  Safe pipeline — never destroys floor pixels
# ─────────────────────────────────────────────

def process_floor_mask(raw_mask, image_bgr, smooth_px=5):
    from scipy.ndimage import binary_fill_holes
    h, w = image_bgr.shape[:2]

    # Fill interior holes
    filled = binary_fill_holes(raw_mask > 0).astype(np.uint8) * 255

    # Remove specks smaller than 0.3% of image
    min_area = max(100, int(0.003 * h * w))
    n_lbls, cc, stats, _ = cv2.connectedComponentsWithStats(filled)
    clean = np.zeros_like(filled)
    for lbl in range(1, n_lbls):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[cc == lbl] = 255

    # Safety: if speck removal wiped everything, use raw
    if clean.sum() == 0:
        print("[WARN] Speck removal emptied mask — using raw segmentation output")
        clean = raw_mask.copy()

    # Gentle boundary smooth (Gaussian, NOT erode)
    if smooth_px > 0:
        ksize    = smooth_px * 2 + 1
        blurred  = cv2.GaussianBlur(clean.astype(np.float32), (ksize, ksize), 0)
        smoothed = (blurred > (0.45 * 255)).astype(np.uint8) * 255

        # Safety: never remove more than 20% of floor pixels
        if (smoothed > 0).sum() < (clean > 0).sum() * 0.80:
            print("[WARN] Smoothing too aggressive — skipping")
            smoothed = clean
        clean = smoothed

    return clean


def refine_with_depth(mask, depth_path, thresh_norm):
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        print(f"[WARN] Depth map not found: {depth_path} — skipping")
        return mask
    depth   = cv2.resize(depth, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    refined = np.where((depth.astype(np.float32)/255.0) < thresh_norm,
                       mask, 0).astype(np.uint8)
    if refined.sum() == 0:
        print("[WARN] Depth refinement emptied mask — skipping. "
              "Try lowering --depth_thresh.")
        return mask
    return refined


# ─────────────────────────────────────────────
#  DEBUG OVERLAY
# ─────────────────────────────────────────────

def save_debug_overlay(image_bgr, floor_mask, seg_map, floor_labels, out_path):
    overlay = image_bgr.copy()
    if floor_mask is not None and floor_mask.sum() > 0:
        overlay[floor_mask > 0] = (
            overlay[floor_mask > 0] * 0.45 + np.array([0, 200, 0]) * 0.55
        ).astype(np.uint8)

    unique, counts = np.unique(seg_map, return_counts=True)
    top10 = sorted(zip(counts, unique), reverse=True)[:10]
    y = 35
    for cnt, lbl in top10:
        name = ADE20K_NAMES.get(int(lbl), f"class_{lbl}")
        tag  = "  <-- FLOOR" if int(lbl) in floor_labels else ""
        col  = (0, 220, 0) if int(lbl) in floor_labels else (220, 220, 220)
        text = f"[{lbl:3d}] {name:<22} {100*cnt/seg_map.size:5.1f}%{tag}"
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, col, 1, cv2.LINE_AA)
        y += 26

    cv2.putText(overlay, "GREEN = floor", (12, overlay.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "GREEN = floor", (12, overlay.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)
    print(f"[DEBUG] Overlay: {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a floor mask via SegFormer segmentation.")
    parser.add_argument("--image",        required=True)
    parser.add_argument("--output",       default="floor_mask.png")
    parser.add_argument("--model",        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--depth",        default=None)
    parser.add_argument("--depth_thresh", type=float, default=40)
    parser.add_argument("--floor_labels", type=int, nargs="+", default=[3])
    parser.add_argument("--smooth_px",    type=int, default=5)
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    check_imports()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load: {args.image}")
    h, w = image_bgr.shape[:2]
    print(f"[INFO] Image: {w}×{h}")
    print(f"[INFO] Floor labels: {args.floor_labels} "
          f"({', '.join(ADE20K_NAMES.get(l, str(l)) for l in args.floor_labels)})")

    seg_map  = run_segformer(image_bgr, args.model)
    raw_mask = labels_to_mask(seg_map, args.floor_labels)
    raw_pct  = 100 * (raw_mask > 0).sum() / (h * w)
    print(f"[INFO] Raw floor coverage: {raw_pct:.1f}%")

    if raw_pct < 1.0:
        print("[WARN] Very little floor detected. Top classes:")
        unique, counts = np.unique(seg_map, return_counts=True)
        for cnt, lbl in sorted(zip(counts, unique), reverse=True)[:8]:
            print(f"       [{lbl:3d}] {ADE20K_NAMES.get(int(lbl),f'class_{lbl}'):<25} "
                  f"{100*cnt/seg_map.size:.1f}%")
        print("       Try: --floor_labels 3 <N>")

    print(f"[INFO] Processing (smooth_px={args.smooth_px})…")
    mask = process_floor_mask(raw_mask, image_bgr, smooth_px=args.smooth_px)

    if args.depth:
        print(f"[INFO] Depth refinement: thresh={args.depth_thresh}")
        mask = refine_with_depth(mask, args.depth, args.depth_thresh / 255.0)

    cv2.imwrite(args.output, mask)
    pct = 100 * (mask > 0).sum() / (h * w)
    print(f"[DONE] Floor mask: {args.output}  ({pct:.1f}%)")

    if pct == 0:
        print("[ERROR] Mask is empty. Try --depth_thresh 10 or remove --depth.")

    if args.debug:
        debug_path = os.path.splitext(args.output)[0] + "_debug.png"
        save_debug_overlay(image_bgr, mask, seg_map, args.floor_labels, debug_path)


if __name__ == "__main__":
    main()