"""Geometric scene analysis: metric depth + surface normals (Metric3D v2) and
gravity estimation (classical vanishing-point). Runs in system Python (no bpy).

Models
------
Metric3D v2 (Hu et al., TPAMI 2024 — arXiv:2404.15506)
    Single forward pass yields geometrically consistent metric depth and surface
    normals. Replaces the earlier two-model setup (Depth Anything V2 + Omnidata).
    Loaded via torch.hub from YvanYin/Metric3D; variant: metric3d_vit_small.

Gravity estimation — classical LSD + RANSAC vanishing-point (no ML model)
    von Gioi et al., "LSD: A Fast Line Segment Detector", TPAMI 2010.
    doi:10.1109/TPAMI.2008.300
"""

import threading

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#  Metric3D v2 — depth + surface normals                                      #
# --------------------------------------------------------------------------- #

_METRIC3D_MODEL = None
_METRIC3D_LOCK = threading.Lock()

# ViT-small canonical input size (height × width)
_M3D_INPUT_SIZE = (616, 1064)
# ImageNet normalisation constants (values in [0, 255] range)
_M3D_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_M3D_STD  = np.array([58.395,  57.12,  57.375],  dtype=np.float32)


def _cuda_available():
    """Return True only if CUDA is both compiled-in and actually usable at runtime.

    torch.cuda.is_available() and torch.cuda.init() both succeed in WSL even
    when no NVIDIA driver is present, because CUDA is lazily initialised and
    the driver is only contacted on the first real kernel execution.  We
    therefore run a tiny GPU op to force that check here.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Force an actual kernel launch — this is where the driver error surfaces.
        t = torch.zeros(1, device="cuda")
        del t
        return True
    except Exception:
        return False


def _patch_metric3d_for_device(model, device):
    """Monkey-patch hardcoded device='cuda' calls inside Metric3D for CPU inference.

    Metric3D's RAFTDepthNormalDPTDecoder5.get_bins() creates tensors with
    device='cuda' hardcoded. We replace it with a version that queries the
    actual device the model lives on, so it works on CPU too.
    """
    import types
    import math as _math

    for module in model.modules():
        if callable(getattr(module, "get_bins", None)) and hasattr(module, "min_val"):
            def _get_bins(self, bins_num):
                import torch as _torch
                _device = next(self.parameters()).device
                vec = _torch.linspace(
                    _math.log(self.min_val), _math.log(self.max_val),
                    bins_num, device=_device,
                )
                return _torch.exp(vec)
            module.get_bins = types.MethodType(_get_bins, module)

        # HourGlassDecoder.create_mesh_grid also defaults to device='cuda'
        if callable(getattr(module, "create_mesh_grid", None)):
            def _create_mesh_grid(self, height, width, batch, device=None, set_buffer=True):
                import torch as _torch
                if device is None:
                    device = next(self.parameters()).device
                xs = _torch.linspace(0, width - 1, width, device=device)
                ys = _torch.linspace(0, height - 1, height, device=device)
                ys, xs = _torch.meshgrid(ys, xs, indexing="ij")
                meshgrid = _torch.stack([xs, ys], dim=-1)          # (H, W, 2)
                meshgrid = meshgrid.unsqueeze(0).expand(batch, -1, -1, -1)
                return meshgrid
            module.create_mesh_grid = types.MethodType(_create_mesh_grid, module)


def _get_metric3d_model():
    """Lazily load and cache the Metric3D v2 ViT-small model (process-local)."""
    global _METRIC3D_MODEL
    if _METRIC3D_MODEL is None:
        with _METRIC3D_LOCK:
            if _METRIC3D_MODEL is None:
                import torch
                model = torch.hub.load(
                    "YvanYin/Metric3D", "metric3d_vit_small", pretrain=True
                )
                device = torch.device("cuda" if _cuda_available() else "cpu")
                model = model.to(device)
                model.eval()
                _patch_metric3d_for_device(model, device)
                _METRIC3D_MODEL = model
    return _METRIC3D_MODEL


def preload_models():
    """Preload Metric3D into memory to reduce first-request latency."""
    _get_metric3d_model()


def estimate_metric3d(image_path):
    """Estimate metric depth and surface normals from a single image.

    Uses Metric3D v2 (ViT-small) in a single forward pass, producing
    geometrically consistent depth and normals.

    Camera intrinsics are not required; a reasonable indoor default
    (fx = fy = max(H, W), principal point at image centre) is used so that
    Metric3D's canonical-space normalisation works correctly.

    Args:
        image_path: Path to the kitchen image.

    Returns:
        dict with keys:
            'depth'   – np.ndarray (H, W)    metric depth in metres.
            'normals' – np.ndarray (H, W, 3) unit surface normals in camera
                        space (X right, Y up, Z out-of-screen), matching the
                        convention used by build_surface_probability_map.
            'fx'      – float  estimated horizontal focal length in pixels
                        (used for depth-based pest scaling).
    """
    import torch

    model = _get_metric3d_model()
    device = next(model.parameters()).device

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    # Default intrinsics: assume ~60 deg FOV for indoor kitchen camera
    fx = fy = float(max(orig_h, orig_w))
    intrinsic = [fx, fy, orig_w / 2.0, orig_h / 2.0]

    # Resize to canonical input size, preserving aspect ratio
    scale = min(_M3D_INPUT_SIZE[0] / orig_h, _M3D_INPUT_SIZE[1] / orig_w)
    rsz_h, rsz_w = int(orig_h * scale), int(orig_w * scale)
    rgb_rsz = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_LINEAR)

    # Scale intrinsics accordingly
    intr_scaled = [intrinsic[0] * scale, intrinsic[1] * scale,
                   intrinsic[2] * scale, intrinsic[3] * scale]

    # Pad to exact canonical size
    pad_h = _M3D_INPUT_SIZE[0] - rsz_h
    pad_w = _M3D_INPUT_SIZE[1] - rsz_w
    rgb_pad = np.pad(rgb_rsz, ((0, pad_h), (0, pad_w), (0, 0)),
                     mode="constant", constant_values=0)

    # Normalise and build tensor
    rgb_norm = (rgb_pad.astype(np.float32) - _M3D_MEAN) / _M3D_STD
    img_tensor = (torch.from_numpy(rgb_norm)
                  .permute(2, 0, 1)
                  .unsqueeze(0)
                  .to(device))

    with torch.no_grad():
        pred_depth, _, output_dict = model.inference({"input": img_tensor})

    # Unpad and restore original resolution — depth
    depth = pred_depth.squeeze().cpu().numpy()
    depth = depth[:rsz_h, :rsz_w]
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Unpad and restore original resolution — normals
    raw_normal = output_dict.get("prediction_normal")
    if raw_normal is not None:
        # Shape: (1, 4, H, W) — channels 0:3 are XYZ, channel 3 is confidence
        normals = raw_normal[0, :3].permute(1, 2, 0).cpu().numpy()
        normals = normals[:rsz_h, :rsz_w]
        normals = cv2.resize(normals, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mag = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= np.clip(mag, 1e-8, None)
    else:
        # Fallback: finite-difference normals from depth
        normals = compute_surface_normals(depth)

    return {"depth": depth, "normals": normals, "fx": fx}


def compute_surface_normals(depth_map):
    """Compute surface normals from a depth map using np.gradient.

    Args:
        depth_map: np.ndarray (H, W) depth in meters.

    Returns:
        normals: np.ndarray (H, W, 3) with unit normals. The Z component
                 (normals[:,:,2]) indicates surface orientation:
                 ~1.0 = flat horizontal, ~0.0 = vertical wall, <0 = ceiling.
    """
    dy, dx = np.gradient(depth_map)
    # Normal = (-dz/dx, -dz/dy, 1), then normalize
    normals = np.stack([-dx, -dy, np.ones_like(depth_map)], axis=-1)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    magnitude = np.clip(magnitude, 1e-8, None)
    normals /= magnitude
    return normals


def save_depth_preview(depth_map, output_path):
    """Save a normalized grayscale preview image for visual inspection."""
    depth = np.array(depth_map, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        preview = np.zeros_like(depth, dtype=np.uint8)
    else:
        valid = depth[finite]
        d_min = float(np.percentile(valid, 2))
        d_max = float(np.percentile(valid, 98))
        if d_max <= d_min:
            d_min = float(valid.min())
            d_max = float(valid.max()) if float(valid.max()) > float(valid.min()) else d_min + 1.0
        depth = np.clip(depth, d_min, d_max)
        norm = (depth - d_min) / max(d_max - d_min, 1e-8)
        # Dark = far, bright = near for easier visual parsing
        preview = (255.0 * (1.0 - norm)).astype(np.uint8)

    Image.fromarray(preview, mode="L").save(output_path)


def save_surface_preview(normals, output_path):
    """Save an RGB preview of surface normals for visual inspection."""
    normals = np.array(normals, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    # Map [-1, 1] -> [0, 255]
    rgb = np.clip((normals + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def save_mask_preview(mask, output_path):
    """Save a placement map image (binary or probabilistic) to grayscale PNG."""
    arr = np.array(mask)
    if arr.dtype == np.bool_:
        img = (arr.astype(np.uint8) * 255)
    else:
        img = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_path)


def compute_depth_placement_score(depth_map):
    """Compute a normalized near-surface score from depth only (near=1, far=0)."""
    depth = np.array(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)

    valid = depth[finite]
    d_min = float(np.percentile(valid, 2))
    d_max = float(np.percentile(valid, 98))
    if d_max <= d_min:
        d_min = float(valid.min())
        d_max = float(valid.max()) if float(valid.max()) > float(valid.min()) else d_min + 1.0

    depth = np.clip(depth, d_min, d_max)
    norm = (depth - d_min) / max(d_max - d_min, 1e-8)
    # Higher score = closer to camera
    score = 1.0 - norm
    return score.astype(np.float32)


def build_placement_mask(normals, nz_threshold):
    """Build a binary mask of pixels where a pest can be placed.

    Args:
        normals: np.ndarray (H, W, 3) surface normals.
        nz_threshold: Minimum Z-component of the normal. Lower values
                      allow steeper surfaces (walls, overhangs).

    Returns:
        mask: np.ndarray (H, W) boolean mask.
    """
    nz = normals[:, :, 2]
    return nz > nz_threshold


def build_depth_placement_mask(depth_map, threshold):
    """Build a binary placement mask from depth only using a near-score threshold."""
    score = compute_depth_placement_score(depth_map)
    return score > float(threshold)


def build_surface_probability_map(normals, nz_threshold, softness=1.0):
    """Build a per-pest probabilistic placement map from predicted normals.

    Formula:
        slope_prob = sigmoid((nz - nz_threshold) / softness)
        coherence = mean_neighbor_cosine_similarity(normals) mapped to [0, 1]
        p = slope_prob * coherence

    This uses all three normal channels via the coherence term, while keeping
    the pest-specific slope threshold behavior through nz.
    """
    normals = np.array(normals, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure unit normals before cosine-similarity coherence.
    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    mag = np.clip(mag, 1e-8, None)
    normals = normals / mag

    nz = normals[:, :, 2]
    nz = np.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
    softness = float(max(softness, 1e-4))
    logits = (nz - float(nz_threshold)) / softness
    logits = np.clip(logits, -30.0, 30.0)
    slope_prob = 1.0 / (1.0 + np.exp(-logits))

    coherence = _normal_coherence_map(normals)
    prob = slope_prob * coherence
    return prob.astype(np.float32)


def _normal_coherence_map(normals):
    """Return local normal coherence in [0,1] using all 3 channels."""
    # Cosine similarity with 4-neighbors (normals are unit length).
    c = np.zeros(normals.shape[:2], dtype=np.float32)
    n = 0

    # Up/down
    sim_ud = np.sum(normals[1:, :, :] * normals[:-1, :, :], axis=-1)
    c[1:, :] += sim_ud
    c[:-1, :] += sim_ud
    n += 2

    # Left/right
    sim_lr = np.sum(normals[:, 1:, :] * normals[:, :-1, :], axis=-1)
    c[:, 1:] += sim_lr
    c[:, :-1] += sim_lr
    n += 2

    c /= float(n)
    # Cosine similarity is in [-1,1]; map to [0,1].
    c = np.clip((c + 1.0) * 0.5, 0.0, 1.0)
    return c.astype(np.float32)


def build_surface_group_masks(normals):
    """Build directional surface probability maps from predicted normals.
    We split visible surfaces into:
      - up:           floor/counters/shelves
      - down:         ceiling/undersides
      - side_left:    side surfaces with normals pointing camera-left
      - side_right:   side surfaces with normals pointing camera-right
      - side_toward:  side surfaces with normals pointing toward camera

    The side classes are built by first detecting "side-ness" (|ny| near 0),
    then distributing that side probability among left/right/toward gates.
    All maps are coherence-weighted so noisy edge regions are suppressed.

    Args:
        normals: (H, W, 3) float32 camera-space unit normals from Metric3D.

    Returns:
        dict with keys "up", "side_left", "side_right", "side_toward", "down".
        Each value is a (H, W) float32 array in [0..1].
    """
    normals = np.array(normals, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    mag = np.clip(mag, 1e-8, None)
    normals = normals / mag

    # Use the Y component (ny) of camera-space normals as the vertical discriminant.
    #
    # Metric3D stores normals with Y pointing DOWN in image space (OpenCV convention),
    # confirmed by the surface-normal colour preview:
    #   floor (world-up)  → ny ≈ -1  → G channel ≈ 0   → purple  ✓
    #   wall  (leftward)  → nx ≈ -1  → R channel ≈ 0   → cyan    ✓
    #   wall  (rightward) → nx ≈ +1  → R channel ≈ 255 → pink    ✓
    #
    # Vertical split:
    #   up:   ny < -0.5   → normal points world-up   (floor / counter / shelf)
    #   down: ny >  0.3   → normal points world-down (ceiling / underside)
    # Horizontal side split:
    #   left/right from nx sign, toward-camera from nz sign.
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]          # camera Y = image-down = world-up when negative
    nz = normals[:, :, 2]
    abs_ny = np.abs(ny)
    coherence = _normal_coherence_map(normals)

    # Tighten surface separation by ~10% (user-requested stricter distinction).
    _STRICT = 1.10
    _UP_CENTER = 0.5 * _STRICT
    _DOWN_CENTER = 0.3 * _STRICT
    _SIDE_BAND = 0.35 / _STRICT
    _DIR_X_CENTER = 0.15 * _STRICT
    _DIR_Z_CENTER = 0.10 * _STRICT

    def _sig(x, center, steepness=11.0):
        logits = np.clip(steepness * (x - center), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    # Up-facing: ny strongly negative (normal aligns with world up = camera -Y).
    up   = (_sig(-ny, center=_UP_CENTER) * coherence).astype(np.float32)

    # Down-facing: ny strongly positive (normal opposes world up).
    down = (_sig(ny,  center=_DOWN_CENTER) * coherence).astype(np.float32)

    # Side-facing total probability: |ny| near zero.
    side_total = _sig(_SIDE_BAND - abs_ny, center=0.0)

    # Split side surfaces into directional components.
    # Positive nz corresponds to normals pointing toward the camera.
    left_gate   = _sig(-nx, center=_DIR_X_CENTER)
    right_gate  = _sig(nx, center=_DIR_X_CENTER)
    toward_gate = _sig(nz, center=_DIR_Z_CENTER)
    gate_sum = np.clip(left_gate + right_gate + toward_gate, 1e-6, None)

    side_left   = (side_total * (left_gate / gate_sum) * coherence).astype(np.float32)
    side_right  = (side_total * (right_gate / gate_sum) * coherence).astype(np.float32)
    side_toward = (side_total * (toward_gate / gate_sum) * coherence).astype(np.float32)

    return {
        "up": up,
        "side_left": side_left,
        "side_right": side_right,
        "side_toward": side_toward,
        "down": down,
    }

def classify_surface_group_at_pixel(normals, py, px):
    """Return dominant surface group at pixel (py, px)."""
    nx = float(normals[py, px, 0])
    ny = float(normals[py, px, 1])
    nz = float(normals[py, px, 2])
    if np.isnan(nx):
        nx = 0.0
    if np.isnan(ny):
        ny = 0.0
    if np.isnan(nz):
        nz = 0.0
    if ny < -0.55:
        return "up"
    if ny > 0.33:
        return "down"
    if nz > 0.165 and abs(nx) < 0.405:
        return "side_toward"
    return "side_right" if nx >= 0.0 else "side_left"

def build_movement_mask(surface_group_masks, spawn_surface, stickiness):
    """Build a blended per-pest movement probability mask.

    Pixels belonging to the pest's spawn surface group receive weight
    proportional to `stickiness`; pixels of other groups get the residual
    weight split evenly.  The result is normalised so the maximum value is 1.

    Args:
        surface_group_masks: dict returned by build_surface_group_masks().
        spawn_surface:       Surface-group key present in surface_group_masks.
        stickiness:          float in [0, 1] — probability of staying on
                             the same surface group per frame.

    Returns:
        (H, W) float32 movement probability map in [0..1].
    """
    groups = list(surface_group_masks.keys())
    if not groups:
        raise ValueError("surface_group_masks cannot be empty")
    if spawn_surface not in surface_group_masks:
        spawn_surface = groups[0]

    same  = surface_group_masks[spawn_surface]
    other_groups = [g for g in groups if g != spawn_surface]
    if other_groups:
        other = sum(surface_group_masks[g] for g in other_groups) / float(len(other_groups))
    else:
        other = np.zeros_like(same, dtype=np.float32)

    # Use (1-stickiness)^2 so non-spawn surfaces are suppressed quadratically.
    # E.g. rat (0.99): other_weight=0.0001 → wall pixels get < 0.1% probability.
    # E.g. cockroach (0.88): other_weight=0.0144 → still visits other surfaces.
    # Apply an extra 10% suppression to increase inter-surface distinction.
    other_weight = ((1.0 - stickiness) ** 2) * 0.90
    mask = same + other_weight * other
    peak = float(mask.max())
    if peak > 1e-6:
        mask = mask / peak
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def save_movement_mask_preview(movement_mask, spawn_surface, output_path):
    """Save the movement probability mask as a grayscale PNG plus a colorized preview.

    The grayscale PNG at *output_path* is what compute_walk loads (values 0-255
    map directly to probabilities 0-1 via PIL .convert("L")).

    A second file with the same name but a ``_color`` suffix is saved for the
    web app.  It uses a red→yellow→green colormap: value 0 → red, 0.5 → yellow,
    1 → green, matching the requested display scale.
    """
    arr = np.clip(movement_mask, 0.0, 1.0)

    # Grayscale PNG for compute_walk (must remain untouched)
    gray = (arr * 255.0).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(output_path)

    # Red-yellow-green colorized preview for the app
    # R: 1 at val=0, stays 1 to val=0.5, then falls to 0 at val=1
    # G: 0 at val=0, rises to 1 at val=0.5, stays 1 to val=1
    r_ch = np.clip(2.0 * (1.0 - arr), 0.0, 1.0)
    g_ch = np.clip(2.0 * arr, 0.0, 1.0)
    b_ch = np.zeros_like(arr)
    color_arr = (np.stack([r_ch, g_ch, b_ch], axis=-1) * 255.0).astype(np.uint8)
    color_path = output_path.replace(".png", "_color.png")
    Image.fromarray(color_arr, mode="RGB").save(color_path)


def sample_pest_positions_from_probability(probability_map, n, plane_width, plane_height):
    """Sample n starting positions from a probability map via weighted pixel sampling."""
    prob = np.array(probability_map, dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    flat = prob.reshape(-1)
    total = float(flat.sum())
    if total <= 1e-8:
        return sample_pest_positions(prob > 0.5, n, plane_width, plane_height)

    weights = flat / total
    h, w = prob.shape
    margin_frac = 0.05
    idxs = np.random.choice(flat.size, size=n, replace=True, p=weights)

    positions = []
    for flat_idx in idxs:
        py = int(flat_idx // w)
        px = int(flat_idx % w)
        nx = px / max(w - 1, 1)
        ny = py / max(h - 1, 1)
        wx = (nx - 0.5) * plane_width
        wy = (0.5 - ny) * plane_height
        x_lim = plane_width / 2.0 * (1.0 - margin_frac)
        y_lim = plane_height / 2.0 * (1.0 - margin_frac)
        wx = float(np.clip(wx, -x_lim, x_lim))
        wy = float(np.clip(wy, -y_lim, y_lim))
        positions.append((wx, wy))

    return positions


def sample_pest_positions(mask, n, plane_width, plane_height):
    """Sample n pest starting positions from valid mask pixels.

    Converts pixel coordinates to world coordinates matching the Blender plane
    (centered at origin, spanning -plane_width/2..+plane_width/2 on X,
    -plane_height/2..+plane_height/2 on Y).

    Args:
        mask: (H, W) boolean placement mask.
        n: Number of positions to sample.
        plane_width: Width of the kitchen plane in world units.
        plane_height: Height of the kitchen plane in world units.

    Returns:
        List of (x, y) tuples in world coordinates.
    """
    valid_ys, valid_xs = np.where(mask)
    if len(valid_ys) == 0:
        # Fallback: if no valid pixels, return center positions
        return [(0.0, 0.0)] * n

    h, w = mask.shape
    margin_frac = 0.05  # stay slightly away from edges

    positions = []
    for _ in range(n):
        idx = np.random.randint(len(valid_ys))
        px, py = valid_xs[idx], valid_ys[idx]
        # Pixel to normalized [0, 1]
        nx = px / max(w - 1, 1)
        ny = py / max(h - 1, 1)
        # Normalized to world coords (centered at origin)
        wx = (nx - 0.5) * plane_width
        wy = (0.5 - ny) * plane_height  # flip Y: image top = +Y in world
        # Clamp to stay within margin
        x_lim = plane_width / 2.0 * (1.0 - margin_frac)
        y_lim = plane_height / 2.0 * (1.0 - margin_frac)
        wx = float(np.clip(wx, -x_lim, x_lim))
        wy = float(np.clip(wy, -y_lim, y_lim))
        positions.append((wx, wy))

    return positions


# --------------------------------------------------------------------------- #
#  Inference strategy                                                          #
# --------------------------------------------------------------------------- #

def compute_inference_strategy():
    """Return 'parallel' or 'sequential' for the three feed-forward models.

    Rules
    -----
    - No CUDA (CPU only)   → 'parallel'   (no VRAM contention)
    - Multiple GPUs        → 'parallel'   (models can use separate devices)
    - Single GPU           → 'sequential' (avoid VRAM contention / context switching)
    """
    try:
        import torch
        if not _cuda_available():
            return "parallel"
        return "parallel" if torch.cuda.device_count() > 1 else "sequential"
    except Exception:
        return "parallel"


# --------------------------------------------------------------------------- #
#  Gravity / camera-up estimation via vertical vanishing point                #
# --------------------------------------------------------------------------- #

def estimate_gravity(image_path):
    """Estimate the gravity (camera-up) direction from a single indoor image.

    Uses a fully classical vanishing-point pipeline — no additional ML model:
      1. Detect line segments with OpenCV's LSD (falls back to Canny+HoughP).
      2. Keep near-vertical segments (<=30 deg from vertical, >=20 px long).
      3. RANSAC to find the dominant vertical vanishing point (VP).
      4. Convert the VP pixel position to a unit up-vector in camera space.

    Camera-space convention (matching Metric3D v2 surface normals):
        X = right,  Y = up,  Z = out-of-screen toward viewer.

    Falls back to [0, 1, 0] (level-camera prior) when no reliable VP is found.

    Args:
        image_path: Path to the kitchen image.

    Returns:
        dict with keys:
            "gravity_cam"  - np.ndarray (3,) unit vector pointing up in camera space.
            "vp"           - (float, float) VP pixel coords, or None.
            "vert_lines"   - list[(x1,y1,x2,y2)] near-vertical segments used.
            "confidence"   - float inlier fraction in [0, 1].
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _gravity_fallback()

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Line detection
    all_lines = _detect_line_segments(gray)

    # 2. Keep near-vertical segments
    vert_lines = []
    for x1, y1, x2, y2 in all_lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue
        # Angle from vertical: 0 deg = perfectly vertical
        angle_from_vert = np.degrees(abs(np.arctan2(abs(x2 - x1), abs(y2 - y1))))
        if angle_from_vert < 30:
            vert_lines.append((float(x1), float(y1), float(x2), float(y2)))

    if len(vert_lines) < 3:
        return {**_gravity_fallback(), "vert_lines": vert_lines}

    # 3. RANSAC vanishing point
    img_diag = float(np.hypot(w, h))
    vp, n_inliers = _ransac_vanishing_point(vert_lines, img_diag)
    confidence = n_inliers / len(vert_lines) if vert_lines else 0.0

    if vp is None or n_inliers < 3 or confidence < 0.25:
        return {**_gravity_fallback(), "vert_lines": vert_lines,
                "confidence": float(confidence)}

    # 4. VP -> camera-space up vector
    # Camera convention: X=right, Y=up, Z=out-of-screen
    # Image convention: x=right, y=DOWN  ->  flip Y
    cx, cy = w / 2.0, h / 2.0
    f = float(max(w, h))          # rough focal-length estimate (~60 deg FOV)
    gx = (vp[0] - cx) / f
    gy = -(vp[1] - cy) / f        # image-Y down -> camera-Y up
    gz = 1.0
    gravity_cam = np.array([gx, gy, gz], dtype=np.float32)
    mag = np.linalg.norm(gravity_cam)
    if mag < 1e-8:
        return {**_gravity_fallback(), "vert_lines": vert_lines,
                "confidence": float(confidence)}
    gravity_cam /= mag

    return {
        "gravity_cam": gravity_cam,
        "vp": vp,
        "vert_lines": vert_lines,
        "confidence": float(confidence),
    }


def save_gravity_preview(image_path, gravity_result, output_path):
    """Save a visualization of the estimated gravity / camera orientation.

    Draws on the image:
      - Detected near-vertical segments extended to full image height (dark green)
        so their convergence toward the vanishing point is visible even when the
        VP is off-screen.
      - The actual detected segments highlighted on top (bright green).
      - Estimated vanishing point when within / near image bounds (red dot).
      - Horizon line through the image centre, perpendicular to the gravity
        direction and accounting for camera pitch and roll (cyan double arrow).
        A level, non-rolled camera produces a perfectly horizontal line; pitch
        tilts it forward/backward; roll tilts it left/right.

    Below the image a dark panel shows numerical values:
      - gravity_cam vector (X right, Y up, Z out-of-screen)
      - pitch  = camera tilt forward/backward (0 deg = level)
      - roll   = camera roll left/right       (0 deg = upright)
      - detection confidence and number of vertical lines used

    Args:
        image_path:     Path to the original kitchen image.
        gravity_result: Dict returned by estimate_gravity().
        output_path:    Destination path for the preview (JPEG or PNG).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()

    gravity_cam = gravity_result["gravity_cam"]
    vp = gravity_result.get("vp")
    vert_lines = gravity_result.get("vert_lines", [])

    # 1. Extend each vertical segment to full image height (shows convergence)
    for x1, y1, x2, y2 in vert_lines:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) > 1e-6:
            t_top = (0 - y1) / dy
            t_bot = (h - 1 - y1) / dy
            ext_x_top = int(round(x1 + t_top * dx))
            ext_x_bot = int(round(x1 + t_bot * dx))
            p_top = (int(np.clip(ext_x_top, -w, 2 * w)), 0)
            p_bot = (int(np.clip(ext_x_bot, -w, 2 * w)), h - 1)
            cv2.line(vis, p_top, p_bot, (0, 130, 0), 1, cv2.LINE_AA)

    # 2. Detected segments on top (bright green)
    for x1, y1, x2, y2 in vert_lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 230, 0), 2, cv2.LINE_AA)

    # 3. Vanishing point if near/within the image
    if vp is not None:
        vpx, vpy = vp
        if -h <= vpy <= 2 * h and -w <= vpx <= 2 * w:
            dx = int(np.clip(vpx, 0, w - 1))
            dy = int(np.clip(vpy, 0, h - 1))
            cv2.circle(vis, (dx, dy), 10, (0, 0, 255), -1)
            cv2.circle(vis, (dx, dy), 12, (255, 255, 255), 2)

    # 4. Horizon line — perpendicular to gravity up, accounts for pitch + roll.
    #    Up direction in image coords: (gravity_cam[0], -gravity_cam[1])
    #    Rotate 90° CW to get horizon direction: (gravity_cam[1], gravity_cam[0])
    hz_dx = float(gravity_cam[1])
    hz_dy = float(gravity_cam[0])
    mag_hz = np.hypot(hz_dx, hz_dy)
    if mag_hz > 1e-6:
        hz_dx /= mag_hz
        hz_dy /= mag_hz
        cx, cy = w / 2.0, h / 2.0
        # Find t where the horizon line exits the image boundary
        t_candidates = []
        for t in ([(-cx) / hz_dx, (w - 1 - cx) / hz_dx] if abs(hz_dx) > 1e-6 else []):
            t_candidates.append(t)
        for t in ([(- cy) / hz_dy, (h - 1 - cy) / hz_dy] if abs(hz_dy) > 1e-6 else []):
            t_candidates.append(t)
        t_pos = min((t for t in t_candidates if t > 0), default=float(w))
        t_neg = max((t for t in t_candidates if t < 0), default=-float(w))
        p1 = (int(np.clip(cx + t_neg * hz_dx, 0, w - 1)),
              int(np.clip(cy + t_neg * hz_dy, 0, h - 1)))
        p2 = (int(np.clip(cx + t_pos * hz_dx, 0, w - 1)),
              int(np.clip(cy + t_pos * hz_dy, 0, h - 1)))
        icx, icy = int(round(cx)), int(round(cy))
        cv2.arrowedLine(vis, (icx, icy), p1, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.04)
        cv2.arrowedLine(vis, (icx, icy), p2, (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.04)

    # 5. Text panel below the image
    panel_h = 52
    panel = np.full((panel_h, w, 3), 25, dtype=np.uint8)
    gx, gy, gz = float(gravity_cam[0]), float(gravity_cam[1]), float(gravity_cam[2])
    # pitch: forward/backward tilt (arctan2 of Z toward Y; 0 = level)
    pitch_deg = float(np.degrees(np.arctan2(gz, gy)))
    # roll: left/right tilt (arctan2 of X toward Y; 0 = upright)
    roll_deg = float(np.degrees(np.arctan2(gx, gy)))
    conf = gravity_result.get("confidence", 0.0)
    vp_str = f"({vp[0]:.0f}, {vp[1]:.0f})" if vp else "off-frame"
    cv2.putText(panel,
                f"gravity_cam = [{gx:+.3f}, {gy:+.3f}, {gz:+.3f}]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(panel,
                f"pitch={pitch_deg:+.1f}deg  roll={roll_deg:+.1f}deg  "
                f"conf={conf:.2f}  lines={len(vert_lines)}  VP={vp_str}",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

    combined = np.vstack([vis, panel])
    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path)


# ---- gravity helpers ------------------------------------------------------- #

def _gravity_fallback():
    """Level-camera prior: gravity points straight up in camera space."""
    return {
        "gravity_cam": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "vp": None,
        "vert_lines": [],
        "confidence": 0.0,
    }


def _detect_line_segments(gray):
    """Detect line segments. Tries LSD first, falls back to Canny+HoughLinesP."""
    try:
        lsd = cv2.createLineSegmentDetector(0)
        raw, _, _, _ = lsd.detect(gray)
        if raw is not None and len(raw) > 0:
            return raw.reshape(-1, 4).tolist()
    except (cv2.error, AttributeError):
        pass
    # Fallback: Canny edges + probabilistic Hough
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                          minLineLength=30, maxLineGap=10)
    if raw is not None:
        return raw.reshape(-1, 4).tolist()
    return []


def _ransac_vanishing_point(lines, img_diag, n_iter=300):
    """RANSAC to find the dominant vanishing point for a set of line segments.

    Inlier criterion: perpendicular distance from the candidate VP to each
    extended line is less than 2% of the image diagonal.

    Returns:
        (vp, n_inliers) -- vp is (float, float) pixel coords, or (None, 0).
    """
    if len(lines) < 2:
        return None, 0

    arr = np.array(lines, dtype=np.float64)
    n = len(arr)
    thresh = img_diag * 0.02
    best_vp, best_n = None, 0

    for _ in range(n_iter):
        i, j = np.random.choice(n, 2, replace=False)
        vp = _intersect_lines(arr[i], arr[j])
        if vp is None:
            continue
        vpx, vpy = vp
        if not (np.isfinite(vpx) and np.isfinite(vpy)):
            continue
        if abs(vpx) > 1e5 or abs(vpy) > 1e5:
            continue
        dists = np.fromiter(
            (_pt_line_dist(vpx, vpy, *seg) for seg in arr),
            dtype=np.float64, count=n,
        )
        n_in = int((dists < thresh).sum())
        if n_in > best_n:
            best_n = n_in
            best_vp = vp

    return best_vp, best_n


def _intersect_lines(l1, l2):
    """Intersection of two infinite lines via homogeneous coordinates.

    Line through (x1,y1)-(x2,y2):  a*x + b*y + c = 0
        a = y1-y2,  b = x2-x1,  c = x1*y2 - x2*y1
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    a2, b2, c2 = y3 - y4, x4 - x3, x3 * y4 - x4 * y3
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return float(x), float(y)


def _pt_line_dist(px, py, x1, y1, x2, y2):
    """Perpendicular distance from (px,py) to the infinite line through (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    denom = np.hypot(dx, dy)
    if denom < 1e-10:
        return float(np.hypot(px - x1, py - y1))
    return abs(dy * (px - x1) - dx * (py - y1)) / denom
