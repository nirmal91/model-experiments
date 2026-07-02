"""
Generate a tiny, fully-synthetic "vision" dataset from scratch.

Why synthetic instead of downloading a real dataset? This sandbox has
restricted network access, and generating our own images means:
  - zero network dependency, fully reproducible
  - we know the exact ground-truth factors (shape, color, position, size)
    that generated each image, so we can later inspect whether the model's
    learned latent space actually captures them (a classic "disentanglement"
    sanity check used with real autoencoders/VAEs)
  - we can generate a *sequence* (same shape moving frame to frame), which
    is exactly the setup a world model / next-frame predictor needs

Produces:
  data/images/*.png       - IMG_COUNT static images, random shape/color/position
  data/sequence/*.png     - a single shape moving in a straight line across
                            SEQ_LEN frames (used later for the world-model step)
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMG_SIZE = 64
IMG_COUNT = 64
SEQ_LEN = 30
SEED = 0

SHAPES = ["circle", "square", "triangle"]
COLORS = [
    (230, 57, 70),   # red
    (69, 123, 157),  # blue
    (42, 157, 143),  # teal
    (233, 196, 106), # yellow
    (155, 93, 229),  # purple
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def draw_shape(draw, shape, cx, cy, r, color):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        pts = [
            (cx, cy - r),
            (cx - r * math.sin(math.radians(60)), cy + r / 2),
            (cx + r * math.sin(math.radians(60)), cy + r / 2),
        ]
        draw.polygon(pts, fill=color)


def make_image(shape, color, cx, cy, r, bg=(245, 245, 245)):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape, cx, cy, r, color)
    return img


def generate_static_images(rng):
    out_dir = DATA_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = IMG_SIZE // 6
    for i in range(IMG_COUNT):
        shape = rng.choice(SHAPES)
        color = rng.choice(COLORS)
        cx = rng.randint(r + 2, IMG_SIZE - r - 2)
        cy = rng.randint(r + 2, IMG_SIZE - r - 2)
        img = make_image(shape, color, cx, cy, r)
        img.save(out_dir / f"img_{i:03d}_{shape}.png")
    print(f"wrote {IMG_COUNT} images to {out_dir}")


def generate_sequence(rng):
    out_dir = DATA_DIR / "sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = IMG_SIZE // 6
    shape = rng.choice(SHAPES)
    color = rng.choice(COLORS)

    # straight-line trajectory across the canvas, bouncing off walls,
    # simple deterministic physics -> a world model should learn to predict this
    x, y = float(r + 2), float(rng.randint(r + 2, IMG_SIZE - r - 2))
    vx, vy = 3.0, rng.choice([-2.0, 2.0])

    for t in range(SEQ_LEN):
        if x - r <= 0 or x + r >= IMG_SIZE:
            vx *= -1
        if y - r <= 0 or y + r >= IMG_SIZE:
            vy *= -1
        x = min(max(x + vx, r), IMG_SIZE - r)
        y = min(max(y + vy, r), IMG_SIZE - r)
        img = make_image(shape, color, int(x), int(y), r)
        img.save(out_dir / f"frame_{t:03d}.png")
    print(f"wrote {SEQ_LEN} sequence frames to {out_dir} (shape={shape})")


if __name__ == "__main__":
    rng = random.Random(SEED)
    generate_static_images(rng)
    generate_sequence(rng)
