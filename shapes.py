"""Generate random blob masks used as brush shapes by generate.py."""

import argparse
import os
import random

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


def main():
    parser = argparse.ArgumentParser(description="Generate random brush masks.")
    parser.add_argument("--count", type=int, default=10, help="number of masks")
    parser.add_argument("--size", type=int, default=400, help="mask canvas size (px)")
    parser.add_argument("--shapes", type=int, default=2, help="blobs per mask")
    parser.add_argument("--out", default="shapes", help="output directory")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        image = generate_random_mask(args.size, args.size, args.shapes)
        image.save(os.path.join(args.out, f"{i}.png"))


if __name__ == "__main__":
    main()
