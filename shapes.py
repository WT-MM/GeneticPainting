"""Generate random brush masks used by generate.py.

Two styles: hard-edged blob masks (rectangles/ellipses) and textured
paint strokes (bundles of wobbly semi-transparent fibers with tapered
ends, similar in spirit to genetic-drawing's scanned brushes).
"""

import argparse
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw


def generate_random_mask(width, height, num_shapes):
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = (0, 0, 0, 255)

    for _ in range(num_shapes):
        shape_type = random.choice(["rectangle", "ellipse", "circle"])
        x1 = random.randint(width // 4, width // 2)
        y1 = random.randint(height // 4, height // 2)
        x2 = random.randint(x1 + 4, min(x1 + width // 4, width - 1))
        y2 = random.randint(y1 + 4, min(y1 + height // 4, height - 1))

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            radius = min(x2 - x1, y2 - y1) // 2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    return image


def generate_stroke_brush(width, height, rng):
    """A textured paint stroke: many low-opacity fibers scattered around a
    curved centerline, tapered at both ends, lightly blurred."""
    acc = np.zeros((height, width), np.float32)
    margin = width // 8
    xs = np.linspace(margin, width - margin, 60)
    t = np.linspace(-1.0, 1.0, 60)
    base = (
        height * (0.5 + rng.uniform(-0.08, 0.08))
        + rng.uniform(-0.15, 0.15) * height * t**2
        + rng.uniform(-0.06, 0.06) * height * t
    )

    body = height * rng.uniform(0.08, 0.2)
    for _ in range(rng.randint(60, 120)):
        offset = rng.gauss(0, body)
        wobble = np.cumsum([rng.gauss(0, 0.5) for _ in range(60)])
        ys = base + offset + wobble
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        layer = np.zeros_like(acc)
        cv2.polylines(layer, [pts], False, 1.0, rng.choice([1, 2, 2, 3]))
        acc += rng.uniform(0.06, 0.25) * layer

    taper = np.sin(np.linspace(0, np.pi, width)) ** 0.5
    acc = np.clip(acc, 0, 1) * taper[None, :]
    acc = cv2.GaussianBlur(acc, (0, 0), 1.2)

    alpha = (np.clip(acc, 0, 1) * 255).astype(np.uint8)
    white = np.full_like(alpha, 255)
    return np.dstack([white, white, white, alpha])


def main():
    parser = argparse.ArgumentParser(description="Generate random brush masks.")
    parser.add_argument("--count", type=int, default=10, help="number of masks")
    parser.add_argument("--size", type=int, default=400, help="mask canvas size (px)")
    parser.add_argument("--shapes", type=int, default=2, help="blobs per mask")
    parser.add_argument("--style", choices=["blob", "stroke"], default="blob",
                        help="hard-edged blobs or textured paint strokes")
    parser.add_argument("--out", default="shapes", help="output directory")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        if args.style == "stroke":
            brush = generate_stroke_brush(args.size, args.size // 2, rng)
            cv2.imwrite(os.path.join(args.out, f"stroke_{i}.png"), brush)
        else:
            image = generate_random_mask(args.size, args.size, args.shapes)
            image.save(os.path.join(args.out, f"{i}.png"))


if __name__ == "__main__":
    main()
