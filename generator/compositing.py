"""2D sprite compositing pipeline — replaces Blender rendering.

Composites RGBA pest sprites onto the background kitchen image frame-by-frame,
using the random-walk trajectory from pest_animation.compute_walk() and sprites
from pest_models.load_sprite(). Writes frame PNGs and a COCO annotations.json.
"""

import io
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from generator.pest_animation import compute_walk
from generator.pest_models import load_sprite
from generator.labeler import save_coco_dataset
from generator.config import RENDER_WIDTH, RENDER_HEIGHT, PLANE_WIDTH, PLANE_HEIGHT


# ── CCTV simulation ────────────────────────────────────────────────────────────

class CCTVSimulator:
    """Simulates security-camera degradations on synthetic video frames.

    Parameters are sampled **once per video** so that every frame in the same
    clip has consistent noise level, brightness, color mode, etc.  This matches
    real CCTV behaviour where a single recording clip has uniform camera settings.

    Motion blur is handled separately per-frame in composite_frames() because
    its strength depends on each pest's instantaneous pixel velocity.

    Usage::

        sim = CCTVSimulator()          # sample once at video start
        frame_rgb = sim.apply(frame_rgb)  # call for every frame
    """

    def __init__(self):
        # Gaussian sensor noise — every video gets some noise
        self.noise_sigma: float = random.uniform(5.0, 20.0)

        # JPEG compression blocking artefacts — every video
        self.jpeg_quality: int = random.randint(60, 80)

        # Random brightness (exposure variation between cameras)
        self.brightness: float = random.uniform(0.7, 1.3)

        # IR / grayscale night-vision mode — 30 % of videos
        self.grayscale: bool = random.random() < 0.30

        # Low-resolution capture then upscale — 20 % of videos
        self.downscale: bool = random.random() < 0.20

    # ------------------------------------------------------------------
    def apply(self, img: Image.Image) -> Image.Image:
        """Apply all consistent CCTV effects to one frame (RGB PIL image)."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1. IR / grayscale mode
        if self.grayscale:
            img = ImageOps.grayscale(img).convert("RGB")

        # 2. Brightness / exposure
        img = ImageEnhance.Brightness(img).enhance(self.brightness)

        # 3. Gaussian sensor noise
        arr = np.asarray(img, dtype=np.float32)
        noise = np.random.normal(0.0, self.noise_sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(arr)

        # 4. Resolution halve → upsample (simulates low-res capture)
        if self.downscale:
            w, h = img.size
            half = img.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
            img = half.resize((w, h), Image.LANCZOS)

        # 5. JPEG compression artefacts (encode + decode in memory)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).copy()

        return img

    def __repr__(self) -> str:
        return (
            f"CCTVSimulator(noise_sigma={self.noise_sigma:.1f}, "
            f"jpeg_quality={self.jpeg_quality}, "
            f"brightness={self.brightness:.2f}, "
            f"grayscale={self.grayscale}, "
            f"downscale={self.downscale})"
        )


# ── Motion blur helpers ────────────────────────────────────────────────────────

def _motion_blur_kernel(dx: float, dy: float, max_kernel: int = 15) -> np.ndarray | None:
    """Build a directional motion-blur convolution kernel.

    Args:
        dx, dy:     Pixel displacement vector for this frame.
        max_kernel: Maximum kernel side length (odd).

    Returns:
        A (k, k) float32 normalised kernel, or None if displacement is too small.
    """
    length = int(np.clip(np.hypot(dx, dy) * 1.5, 0, max_kernel))
    if length < 3:
        return None
    if length % 2 == 0:
        length += 1

    k = np.zeros((length, length), dtype=np.float32)
    cx = cy = length // 2
    angle = math.atan2(dy, dx)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    for i in range(length):
        t = (i - cx) / max(cx, 1)
        xi = int(round(cx + t * cos_a * cx))
        yi = int(round(cy + t * sin_a * cy))
        if 0 <= xi < length and 0 <= yi < length:
            k[yi, xi] = 1.0

    total = k.sum()
    if total < 1e-6:
        return None
    return k / total


def apply_motion_blur_to_sprite(
    sprite: Image.Image,
    dx_px: float,
    dy_px: float,
) -> Image.Image:
    """Apply directional motion blur to an RGBA sprite.

    RGB channels are blurred along the velocity vector; alpha is blurred
    with the same kernel to give a natural motion-ghost transparency fall-off.

    Args:
        sprite:  RGBA PIL image of the pest sprite.
        dx_px:   Horizontal pixel displacement this frame.
        dy_px:   Vertical pixel displacement this frame.

    Returns:
        Motion-blurred RGBA PIL image (same size as input).
    """
    kernel = _motion_blur_kernel(dx_px, dy_px)
    if kernel is None:
        return sprite

    try:
        from scipy.ndimage import convolve  # type: ignore
    except ImportError:
        return sprite  # skip gracefully if scipy unavailable

    if sprite.mode != "RGBA":
        sprite = sprite.convert("RGBA")

    arr = np.asarray(sprite, dtype=np.float32)  # (H, W, 4)
    out = np.empty_like(arr)
    for ch in range(4):
        out[..., ch] = np.clip(convolve(arr[..., ch], kernel, mode="constant"), 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


_CATEGORY_MAP = {"mouse": 1, "rat": 2, "cockroach": 3}
_CATEGORIES = [
    {"id": 1, "name": "mouse",     "supercategory": "pest"},
    {"id": 2, "name": "rat",       "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"},
]


def composite_frames(
    image_path,
    pest_configs,
    frames_dir,
    labels_dir,
    num_frames,
    sprites_dir,
    render_width=RENDER_WIDTH,
    render_height=RENDER_HEIGHT,
    plane_width=PLANE_WIDTH,
    plane_height=PLANE_HEIGHT,
    depth_map=None,
    fps=10,
    surface_group_masks=None,
    normals=None,
    save_mask_previews=True,
    frame_format="png",
    save_every_n=1,
    keep_full_annotations=False,
    cctv_sim=None,
):
    """Render all frames by compositing pest sprites onto the background image.

    For each frame i in [1..num_frames]:
      1. Copy the background image (resized to render_width x render_height).
      2. For each pest: look up (wx, wy, angle_rad) from its walk trajectory,
         resize sprite, rotate, and alpha-composite onto the frame.
      3. Save as frames_dir/frame_{i:04d}.<ext>.

    Also writes a COCO annotations.json to labels_dir.

    Args:
        image_path:    Path to the kitchen background image.
        pest_configs:  List of pest config dicts from pipeline.py. Each must have:
                           "type"                – pest type string
                           "start_position"      – [wx, wy] world units
                           "placement_mask_path" – path to movement mask PNG
                           "surface_mask_paths"  – optional {surface_group: mask_png_path}
                           "params": {
                               "speed"             – float (wu/frame)
                               "max_speed_wps"     – float (wu/s, for depth-aware cap)
                               "focal_length_px"   – float
                               "forward_axis"      – str
                               "blender_scale"     – float (world-unit body length)
                               "sprite_width_scale"  – optional float (default 1.0)
                               "sprite_height_scale" – optional float (default 1.0)
                           }
        frames_dir:    Directory to write frame_XXXX.<ext> files into.
        labels_dir:    Directory to write annotations.json into.
        num_frames:    Number of frames to render.
        sprites_dir:   Root directory for per-pest-type sprite PNG files.
        render_width:  Output frame width in pixels.
        render_height: Output frame height in pixels.
        plane_width:   World-space width of the scene.
        plane_height:  World-space height of the scene.
        depth_map:     Optional (H, W) float32 depth array in metres.
        fps:           Frames per second (used for depth-aware speed cap).
        save_mask_previews: Whether to save per-pest dynamic mask preview PNGs.
        frame_format: Output frame format. Supported: png, jpg/jpeg, webp.
        save_every_n: Persist frame 1 and then every Nth rendered frame while
            still simulating the full trajectory. Saved frame numbering matches
            the original timeline.
        keep_full_annotations: If True, write COCO image/annotation entries for
            all simulated frames even when only a sparse subset of frame images
            is persisted on disk.
        cctv_sim: Optional CCTVSimulator instance. When provided, each saved
            frame is passed through it (noise, compression, brightness, etc.)
            before writing to disk. Motion blur is also applied to each sprite
            proportional to its pixel velocity. Pass None to skip simulation.
    """
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    frame_ext = _normalize_frame_format(frame_format)
    save_every_n = max(1, int(save_every_n or 1))

    # Load and resize background once
    bg = Image.open(image_path).convert("RGBA")
    bg = bg.resize((render_width, render_height), Image.LANCZOS)

    # Pre-compute walk trajectories and sprites for all pests
    pest_data = []
    for pest_cfg in pest_configs:
        pest_type     = pest_cfg["type"]
        params        = pest_cfg["params"]
        speed            = float(params.get("speed", 0.08))
        fwd_axis         = params.get("forward_axis", "X")
        blender_scale    = float(params.get("blender_scale", 0.12))
        max_step_world   = params.get("max_step_world")
        if max_step_world is not None:
            max_step_world = float(max_step_world)
        base_speed_wps   = params.get("base_speed_wps")
        if base_speed_wps is not None:
            base_speed_wps = float(base_speed_wps)
        max_speed_wps    = params.get("max_speed_wps")
        if max_speed_wps is not None:
            max_speed_wps = float(max_speed_wps)
        focal_length_px  = params.get("focal_length_px")
        if focal_length_px is not None:
            focal_length_px = float(focal_length_px)
        max_turn_deg     = params.get("max_turn_deg", 2.0)
        stickiness       = float(params.get("surface_stickiness", 0.97))
        pause_chance     = float(params.get("pause_chance", 0.008))
        sprite_width_scale  = float(params.get("sprite_width_scale", 1.0))
        sprite_height_scale = float(params.get("sprite_height_scale", 1.0))

        walk = compute_walk(
            num_frames=num_frames,
            plane_width=plane_width,
            plane_height=plane_height,
            speed=speed,
            start_position=pest_cfg.get("start_position"),
            placement_mask_path=pest_cfg.get("placement_mask_path"),
            placement_mask_array=pest_cfg.get("placement_mask_array"),
            surface_mask_paths=pest_cfg.get("surface_mask_paths"),
            surface_mask_arrays=pest_cfg.get("surface_mask_arrays"),
            forward_axis=fwd_axis,
            max_step_world=max_step_world,
            depth_map=depth_map,
            focal_length_px=focal_length_px,
            base_speed_wps=base_speed_wps,
            max_speed_wps=max_speed_wps,
            fps=fps,
            render_width=render_width,
            render_height=render_height,
            max_turn_deg=max_turn_deg,
            surface_group_masks=surface_group_masks,
            normals=normals,
            surface_stickiness=stickiness,
            pause_chance=pause_chance,
        )

        sprite   = load_sprite(pest_type, sprites_dir)
        pixel_w, pixel_h = _compute_pixel_size(
            sprite, blender_scale, plane_width, render_width,
            width_scale=sprite_width_scale,
            height_scale=sprite_height_scale,
        )

        pest_data.append((pest_type, sprite, walk, pixel_w, pixel_h))

    # Pre-load movement masks as numpy arrays for per-frame overlay rendering.
    _DOT_COLORS = {"mouse": (0, 220, 0), "rat": (255, 120, 0), "cockroach": (255, 30, 30)}
    mask_arrays = []
    surface_groups = tuple(surface_group_masks.keys()) if surface_group_masks is not None else ()
    for pest_cfg in pest_configs:
        placement_arr = pest_cfg.get("placement_mask_array")
        if isinstance(placement_arr, np.ndarray):
            arr = np.clip(np.asarray(placement_arr, dtype=np.float32), 0.0, 1.0)
            m = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize(
                (render_width, render_height), Image.LANCZOS
            )
            mask_arrays.append(np.array(m, dtype=np.float32) / 255.0)
            continue
        mpath = pest_cfg.get("placement_mask_path")
        if mpath and os.path.exists(mpath):
            m = Image.open(mpath).convert("L").resize(
                (render_width, render_height), Image.LANCZOS)
            mask_arrays.append(np.array(m, dtype=np.float32) / 255.0)
        else:
            mask_arrays.append(np.ones((render_height, render_width), dtype=np.float32))

    # Pre-compute per-pest dynamic mask arrays (one per surface group) resized to
    # render dimensions. Used for per-frame "pest vision" mask preview.
    _viz_dynamic_masks = []
    for pest_cfg in pest_configs:
        surface_mask_arrays = pest_cfg.get("surface_mask_arrays") or {}
        if isinstance(surface_mask_arrays, dict) and surface_mask_arrays:
            dyn_viz = {}
            for surf, arr in surface_mask_arrays.items():
                try:
                    arr = np.asarray(arr, dtype=np.float32)
                except Exception:
                    continue
                if arr.ndim != 2:
                    continue
                arr = np.clip(arr, 0.0, 1.0)
                m_pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize(
                    (render_width, render_height), Image.LANCZOS
                )
                dyn_viz[surf] = np.array(m_pil, dtype=np.float32) / 255.0
            _viz_dynamic_masks.append(dyn_viz if dyn_viz else None)
            continue

        surface_mask_paths = pest_cfg.get("surface_mask_paths") or {}
        if surface_mask_paths:
            dyn_viz = {}
            for surf, mpath in surface_mask_paths.items():
                if not mpath or not os.path.exists(mpath):
                    continue
                m_pil = Image.open(mpath).convert("L").resize(
                    (render_width, render_height), Image.LANCZOS
                )
                dyn_viz[surf] = np.array(m_pil, dtype=np.float32) / 255.0
            _viz_dynamic_masks.append(dyn_viz if dyn_viz else None)
        elif surface_group_masks is not None and surface_groups:
            stickiness = float(pest_cfg["params"].get("surface_stickiness", 0.97))
            ow = (1.0 - stickiness) ** 2
            dyn_viz = {}
            for surf in surface_groups:
                same  = surface_group_masks[surf]
                other_groups = [g for g in surface_groups if g != surf]
                if other_groups:
                    other = sum(surface_group_masks[g] for g in other_groups) / float(len(other_groups))
                else:
                    other = np.zeros_like(same, dtype=np.float32)
                m     = same + ow * other
                peak  = float(m.max())
                m_norm = (m / peak if peak > 1e-6 else m).astype(np.float32)
                m_pil  = Image.fromarray((m_norm * 255).astype(np.uint8))
                m_pil  = m_pil.resize((render_width, render_height), Image.LANCZOS)
                dyn_viz[surf] = np.array(m_pil, dtype=np.float32) / 255.0
            _viz_dynamic_masks.append(dyn_viz)
        else:
            _viz_dynamic_masks.append(None)

    # Render each frame
    coco_images      = []
    coco_annotations = []
    ann_id = 1

    for frame_idx in range(num_frames):
        frame_num = frame_idx + 1
        persist_frame = frame_num == 1 or frame_num % save_every_n == 0

        frame_img = bg.copy()

        for pest_type, sprite, walk, pixel_w, pixel_h in pest_data:
            wx, wy, angle_rad, _surface = walk[frame_idx]

            paste_cx, paste_cy = _world_to_pixel(
                wx, wy, render_width, render_height, plane_width, plane_height
            )

            resized = sprite.resize((pixel_w, pixel_h), Image.LANCZOS)

            # PIL.rotate is counter-clockwise; atan2 is also CCW — signs match.
            angle_deg = math.degrees(angle_rad)
            rotated = resized.rotate(
                angle_deg,
                resample=Image.BICUBIC,
                expand=True,
            )
            rotated = _sharpen_rgba(rotated)

            # Motion blur proportional to pixel velocity (CCTV simulation).
            if cctv_sim is not None and frame_idx + 1 < num_frames:
                nx, ny, _, _ = walk[frame_idx + 1]
                ncx, ncy = _world_to_pixel(
                    nx, ny, render_width, render_height, plane_width, plane_height
                )
                dx_px = float(ncx - paste_cx)
                dy_px = float(ncy - paste_cy)
                rotated = apply_motion_blur_to_sprite(rotated, dx_px, dy_px)

            paste_x = paste_cx - rotated.width  // 2
            paste_y = paste_cy - rotated.height // 2

            # Alpha-composite (rotated is RGBA; expand=True gives transparent corners)
            frame_img.paste(rotated, (paste_x, paste_y), rotated)

            # COCO bbox from rotated sprite bounds, clamped to image
            bbox_x  = max(0, paste_x)
            bbox_y  = max(0, paste_y)
            bbox_x2 = min(render_width,  paste_x + rotated.width)
            bbox_y2 = min(render_height, paste_y + rotated.height)
            bbox_w  = bbox_x2 - bbox_x
            bbox_h  = bbox_y2 - bbox_y

            if bbox_w > 0 and bbox_h > 0:
                if not persist_frame and not keep_full_annotations:
                    continue
                coco_annotations.append({
                    "id":          ann_id,
                    "image_id":    frame_num,
                    "category_id": _CATEGORY_MAP.get(pest_type, 0),
                    "bbox":        [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area":        bbox_w * bbox_h,
                    "iscrowd":     0,
                })
                ann_id += 1

        if persist_frame:
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:04d}.{frame_ext}")
            out_rgb = frame_img.convert("RGB")
            if cctv_sim is not None:
                out_rgb = cctv_sim.apply(out_rgb)
            _save_frame_image(out_rgb, frame_path, frame_ext)

        if persist_frame and save_mask_previews:
            # --- Per-pest dynamic "pest vision" mask preview ---
            # Each pest gets its own preview: the movement probability mask for the
            # surface it currently stands on, with a dot showing its position.
            # The mask switches automatically when the pest crosses to a new surface.
            for pest_idx, ((ptype, _sprite, pwalk, pw, ph), dyn_masks, mask_arr) in enumerate(
                zip(pest_data, _viz_dynamic_masks, mask_arrays)
            ):
                wx, wy, _, surface = pwalk[frame_idx]
                active_viz = dyn_masks.get(surface, mask_arr) if dyn_masks is not None else mask_arr
                gray_u8 = (np.clip(active_viz, 0.0, 1.0) * 255).astype(np.uint8)
                pest_mask_img = Image.fromarray(
                    np.stack([gray_u8, gray_u8, gray_u8], axis=-1), mode="RGB"
                )
                draw = ImageDraw.Draw(pest_mask_img)
                pcx, pcy = _world_to_pixel(
                    wx, wy, render_width, render_height, plane_width, plane_height
                )
                r     = max(pw, ph) // 2 + 6
                color = _DOT_COLORS.get(ptype, (255, 255, 0))
                draw.ellipse([pcx - r, pcy - r, pcx + r, pcy + r], outline=color, width=3)
                draw.ellipse([pcx - 5, pcy - 5, pcx + 5, pcy + 5], fill=color)
                pest_mask_img.save(
                    os.path.join(frames_dir, f"mask_preview_pest{pest_idx}_{ptype}_{frame_num:04d}.png")
                )

        if persist_frame or keep_full_annotations:
            coco_images.append({
                "id":        frame_num,
                "file_name": f"frame_{frame_num:04d}.{frame_ext}",
                "width":     render_width,
                "height":    render_height,
            })

    save_coco_dataset(
        coco_images,
        coco_annotations,
        _CATEGORIES,
        os.path.join(labels_dir, "annotations.json"),
    )


def _world_to_pixel(wx, wy, render_width, render_height, plane_width, plane_height):
    """Convert world-space coordinates to pixel coordinates.

    World origin (0, 0) = centre of image.
    World (+plane_width/2, +plane_height/2) = top-right corner.
    Pixel (0, 0) = top-left corner.
    """
    px = int((wx / plane_width  + 0.5) * render_width)
    py = int((0.5 - wy / plane_height) * render_height)
    px = max(0, min(render_width  - 1, px))
    py = max(0, min(render_height - 1, py))
    return px, py


def _compute_pixel_size(
    sprite,
    blender_scale,
    plane_width,
    render_width,
    width_scale=1.0,
    height_scale=1.0,
):
    """Compute target pixel width and height for a sprite.

    blender_scale is the pest body length in world units.
    pixel_w = blender_scale * render_width / plane_width.
    pixel_h preserves the sprite's aspect ratio before optional axis scales.
    Both dimensions are clamped to at least 4 pixels.
    """
    width_scale = max(0.1, float(width_scale))
    height_scale = max(0.1, float(height_scale))

    base_w = max(4, int(blender_scale * render_width / plane_width))
    sw, sh  = sprite.size
    base_h = max(4, int(base_w * sh / max(sw, 1)))
    pixel_w = max(4, int(base_w * width_scale))
    pixel_h = max(4, int(base_h * height_scale))
    return pixel_w, pixel_h


def _normalize_frame_format(frame_format):
    fmt = str(frame_format or "png").strip().lower()
    if fmt == "jpeg":
        return "jpg"
    if fmt not in {"png", "jpg", "webp"}:
        return "png"
    return fmt


def _save_frame_image(image_rgb, frame_path, frame_ext):
    """Save RGB frame with format-specific settings."""
    if frame_ext == "webp":
        # Lossless WebP: materially smaller than PNG with no quality loss.
        image_rgb.save(frame_path, format="WEBP", lossless=True, quality=100, method=6)
        return
    if frame_ext == "jpg":
        image_rgb.save(frame_path, format="JPEG", quality=95, subsampling=0)
        return
    image_rgb.save(frame_path, format="PNG")


def _sharpen_rgba(img):
    """Sharpen RGB detail while preserving alpha edges."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=170, threshold=2))
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.15)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out
