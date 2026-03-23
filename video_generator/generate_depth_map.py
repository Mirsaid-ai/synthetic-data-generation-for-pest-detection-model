"""
generate_depth_map.py
=====================
Depth estimation using MiDaS (DPT_Large / DPT_Hybrid / MiDaS_small).

Usage:
    python generate_depth_map.py --image kitchen.png
    python generate_depth_map.py --image kitchen.jpg --output kitchen_depth.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Depth estimation using MiDaS")
    parser.add_argument("--image",  "-i", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save depth map (default: <image>_depth.png)")
    parser.add_argument("--model",  type=str, default="DPT_Large",
                        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                        help="MiDaS model type (default: DPT_Large)")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.image).with_suffix("")) + "_depth.png"

    device = get_device()
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading MiDaS model: {args.model}")
    midas = torch.hub.load("intel-isl/MiDaS", args.model)
    midas.eval()
    midas.to(device)

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if args.model in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)

    while input_batch.dim() < 4:
        input_batch = input_batch.unsqueeze(0)
    while input_batch.dim() > 4:
        input_batch = input_batch.squeeze(0)

    input_batch = input_batch.to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    cv2.imwrite(output_path, (depth * 255).astype(np.uint8))
    print(f"[DONE] Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()