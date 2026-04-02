"""Generator configuration — pure Python, no bpy dependency."""

import os


# Render settings
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
FPS = 10
NUM_FRAMES = 300

# Output directories (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
# Preview/test-tab outputs (real generator writes to outputs/train|test/* explicitly).
PREVIEW_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "preview")
FRAMES_DIR = os.path.join(PREVIEW_OUTPUT_DIR, "frames")
VIDEOS_DIR = os.path.join(PREVIEW_OUTPUT_DIR, "videos")
LABELS_DIR = os.path.join(PREVIEW_OUTPUT_DIR, "labels")

# Sprite directories — place RGBA PNG files here to override procedural fallback.
# Resolution order: sprites/{pest_type}/*.png → procedural PIL ellipse.
SPRITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")

# Pest parameters
PEST_TYPES = ["mouse", "rat", "cockroach"]

PEST_PARAMS = {
    "mouse": {
        "body_scale": (0.12, 0.06, 0.05),
        "head_scale": (0.04, 0.035, 0.035),
        "head_offset": (0.13, 0.0, 0.01),
        "color": (0.45, 0.35, 0.25, 1.0),
        # Speeds in world units/second (≈ m/s for a ~2 m wide kitchen scene).
        # base_speed_wps: typical walking pace; max_speed_wps: sprint cap.
        # Sources: mouse 0.25–1.25 m/s walking, up to ~3.6 m/s sprint.
        "base_speed_wps": 0.4,
        "max_speed_wps": 1.5,
        # Spawn probability by surface orientation group.
        # Groups:
        #   "up"          floor / counter / shelf
        #   "side_left"   wall whose normal points camera-left
        #   "side_right"  wall whose normal points camera-right
        #   "side_toward" wall/cabinet face whose normal points toward camera
        #   "down"        ceiling / underside
        "spawn_probs": {
            "up": 1.00,
            "side_left": 0.00,
            "side_right": 0.00,
            "side_toward": 0.00,
            "down": 0.00,
        },
        # Probability of staying on the same surface group per movement frame.
        "surface_stickiness": 0.95,
        # Per-frame pause trigger probability.
        "pause_chance": 0.12,
        # Stretch mouse sprite width by 10% while keeping height unchanged.
        "sprite_width_scale": 1.10,
        "sprite_height_scale": 1.00,
        "max_turn_deg": 3.0,
    },
    "rat": {
        "body_scale": (0.18, 0.09, 0.07),
        "head_scale": (0.06, 0.05, 0.05),
        "head_offset": (0.19, 0.0, 0.015),
        "color": (0.35, 0.25, 0.18, 1.0),
        # Sources: rat 0.3–0.8 m/s walking, max ~2.7 m/s.
        "base_speed_wps": 0.3,
        "max_speed_wps": 1.0,
        # Rat is heavy — almost exclusively floor/counter bound.
        "spawn_probs": {
            "up": 1.00,
            "side_left": 0.00,
            "side_right": 0.00,
            "side_toward": 0.00,
            "down": 0.00,
        },
        # Per-frame pause trigger probability.
        "pause_chance": 0.20,
        # Rat sprite axis scaling relative to original baseline.
        # length (forward axis) = 0.8x, width (cross axis) = 0.7x.
        "sprite_width_scale": 0.80,
        "sprite_height_scale": 0.70,
        "surface_stickiness": 0.99,
        "max_turn_deg": 2.0,
    },
    "cockroach": {
        "body_scale": (0.08, 0.04, 0.015),
        "head_scale": None,  # no separate head
        "head_offset": None,
        "color": (0.25, 0.12, 0.05, 1.0),
        # Sources: cockroach ~0.05–0.15 m/s walking, max ~1.5 m/s (American cockroach).
        "base_speed_wps": 0.6,
        "max_speed_wps": 1.5,
        # Cockroaches can walk on any surface — well-known wall/ceiling climbers.
        "spawn_probs": {
            "up": 0.40,
            "side_left": 0.13,
            "side_right": 0.13,
            "side_toward": 0.12,
            "down": 0.22,
        },
        # Per-frame pause trigger probability.
        "pause_chance": 0.09,
        # Brighten cockroach movement masks by 10% (clipped to [0,1]).
        "movement_mask_brightness": 1.10,
        "surface_stickiness": 0.75,
        "max_turn_deg": 8.0,
    },
}

# Real physical body lengths (metres) used for depth-based scale calculation.
# Body length = longest horizontal dimension of the animal.
PEST_REAL_SIZES_M = {
    "mouse": 0.088,      # +10% vs baseline (~8.8 cm body, excluding tail)
    "rat": 0.18,         # -10% vs baseline (~18 cm body, excluding tail)
    "cockroach": 0.04,   # ~4 cm body
}

# The sprite's local axis that points toward the head (forward direction).
# Used to orient the pest along its movement vector each frame.
# Procedural sprites have the head drawn on the right (+X), so forward_axis = "X".
# Valid values: "X", "-X", "Y", "-Y"
PEST_FORWARD_AXIS = {
    "mouse": "X",
    "rat": "X",
    "cockroach": "X",
}

# Number of pests to place per scene
MIN_PESTS = 1
MAX_PESTS = 3

# Kitchen plane dimensions (world units)
PLANE_WIDTH = 2.0
PLANE_HEIGHT = 1.5
