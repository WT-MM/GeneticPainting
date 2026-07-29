"""Genetic painting: approximate a reference image by layering brush strokes.

Each stroke is evolved with a small genetic algorithm (random population,
selection on how much the stroke reduces error vs. the reference, mutation),
then alpha-composited onto the canvas. Stroke sizes anneal from coarse to
fine so early strokes block in large color regions and later strokes add
detail. Stroke positions are sampled from the current error map so effort
goes where the painting is still wrong.
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np


class GeneticPainting:
    def __init__(self, img, shapes, seed=0, max_dim=1200, canvas="mean", init_canvas=None,
                 edge_weight=1.0):
        reference = cv2.imread(img)
        if reference is None:
            raise FileNotFoundError(f"could not read reference image: {img}")

        scale = min(1.0, max_dim / max(reference.shape[:2]))
        if scale < 1.0:
            reference = cv2.resize(
                reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        self.reference = reference.astype(np.float32)
        self.h, self.w = reference.shape[:2]

        random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Brushes are stored as float masks in [0, 1], cropped to their
        # bounding box. Soft (textured) alpha is preserved. A brush PNG's
        # alpha channel is used when present; otherwise its luminance is
        # the mask (auto-inverted if the image has a bright background).
        self.brushes = self._load_brushes(shapes)
        if not self.brushes:
            raise ValueError("no usable brush shapes found")

        if init_canvas is not None:
            prev = cv2.imread(init_canvas)
            if prev is None:
                raise FileNotFoundError(f"could not read init canvas: {init_canvas}")
            if prev.shape[:2] != (self.h, self.w):
                prev = cv2.resize(prev, (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.canvas = prev.astype(np.float32)
        else:
            if canvas == "mean":
                start = self.reference.mean(axis=(0, 1))
            else:
                start = np.zeros(3, np.float32)
            self.canvas = np.full_like(self.reference, start)

        # Perceptual weighting: plain L1 error underweights small, crisp
        # features (a pale moon on pale clouds is numerically invisible),
        # so per-pixel error is scaled up near edges of the reference.
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 3)
        hi = np.percentile(grad, 95)
        self.weight_map = 1.0 + edge_weight * np.clip(grad / max(hi, 1e-6), 0, 1)

        # Local edge orientation (degrees) for gradient-aligned strokes:
        # new strokes start roughly parallel to nearby edges, which reads
        # as "strokes following the form". getRotationMatrix2D rotates
        # counter-clockwise in display coordinates while image y points
        # down, hence 90 - angle rather than angle + 90. Alignment is only
        # trusted where there is an actual gradient — in flat regions the
        # angle is meaningless and strokes stay random.
        sx = cv2.GaussianBlur(gx, (0, 0), 6)
        sy = cv2.GaussianBlur(gy, (0, 0), 6)
        self.orientation = 90.0 - np.degrees(np.arctan2(sy, sx))
        self.oriented = cv2.magnitude(sx, sy) > 1.0

        # Per-pixel (weighted) error, kept in sync with the canvas.
        self.error_map = (
            np.abs(self.reference - self.canvas).sum(axis=2) * self.weight_map
        )

    @staticmethod
    def _load_brushes(paths):
        """Load brush images as float masks in [0, 1], cropped to their
        bounding box. The alpha channel is the mask when present;
        otherwise luminance (auto-inverted for bright backgrounds)."""
        brushes = []
        for path in paths:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            scale_to_unit = (
                float(np.iinfo(im.dtype).max) if np.issubdtype(im.dtype, np.integer)
                else 1.0
            )
            if im.ndim == 3 and im.shape[2] == 4:
                mask = im[:, :, 3].astype(np.float32) / scale_to_unit
            else:
                gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                mask = gray.astype(np.float32) / scale_to_unit
                border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
                if np.median(border) > 0.5:
                    mask = 1.0 - mask
                # Luminance masks come from scans/photos: kill residual
                # background noise or every stroke drags a faint
                # rectangular wash along with it.
                mask[mask < 0.05] = 0.0
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0:
                continue
            brushes.append(mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
        return brushes

    # ------------------------------------------------------------------
    # Stroke rendering

    def render_mask(self, stroke):
        """Rasterize a stroke's brush mask and return (mask, x0, y0) where
        (x0, y0) is the top-left corner of the mask on the canvas, or None
        if the stroke lands entirely off-canvas."""
        brush = self.brushes[stroke["brush"]]
        bh, bw = brush.shape
        s = stroke["size"] / max(bh, bw)
        mask = cv2.resize(
            brush, (max(1, int(bw * s)), max(1, int(bh * s))),
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
        )

        # Rotate with an expanded output so nothing gets clipped.
        mh, mw = mask.shape
        m = cv2.getRotationMatrix2D((mw / 2, mh / 2), stroke["angle"], 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw = int(mh * sin + mw * cos)
        nh = int(mh * cos + mw * sin)
        m[0, 2] += nw / 2 - mw / 2
        m[1, 2] += nh / 2 - mh / 2
        mask = cv2.warpAffine(mask, m, (nw, nh), flags=cv2.INTER_LINEAR)

        x0 = int(round(stroke["cx"] - nw / 2))
        y0 = int(round(stroke["cy"] - nh / 2))

        # Clip to the canvas.
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(self.w, x0 + nw), min(self.h, y0 + nh)
        if cx0 >= cx1 or cy0 >= cy1:
            return None
        mask = mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        if mask.max() <= 0:
            return None
        return mask, cx0, cy0

    def stroke_delta(self, stroke):
        """Error change if this stroke were painted (negative = improvement).

        Also fills in the stroke's color (mean reference color under the
        brush footprint plus the stroke's evolved jitter) as a side effect.
        """
        rendered = self.render_mask(stroke)
        if rendered is None:
            return np.inf
        mask, x0, y0 = rendered
        mh, mw = mask.shape

        ref = self.reference[y0 : y0 + mh, x0 : x0 + mw]
        cur = self.canvas[y0 : y0 + mh, x0 : x0 + mw]

        weight = (mask * stroke["opacity"])[..., None]
        total = weight.sum()
        if total <= 0:
            return np.inf
        color = (ref * weight).sum(axis=(0, 1)) / total

        # Blend toward the color at the stroke center so small bright/dark
        # features (moon, lit windows) aren't averaged away by a footprint
        # that is mostly background.
        b = stroke["center_bias"]
        if b > 0:
            px = int(np.clip(stroke["cx"], 2, self.w - 3))
            py = int(np.clip(stroke["cy"], 2, self.h - 3))
            center = self.reference[py - 2 : py + 3, px - 2 : px + 3].mean(axis=(0, 1))
            color = (1 - b) * color + b * center
        color = np.clip(color + stroke["jitter"], 0, 255)
        stroke["color"] = color

        new = cur * (1 - weight) + color * weight
        w = self.weight_map[y0 : y0 + mh, x0 : x0 + mw]
        old_err = (np.abs(ref - cur).sum(axis=2) * w).sum()
        new_err = (np.abs(ref - new).sum(axis=2) * w).sum()
        return new_err - old_err

    def commit(self, stroke):
        mask, x0, y0 = self.render_mask(stroke)
        mh, mw = mask.shape
        weight = (mask * stroke["opacity"])[..., None]
        region = np.s_[y0 : y0 + mh, x0 : x0 + mw]
        self.canvas[region] = self.canvas[region] * (1 - weight) + stroke["color"] * weight
        self.error_map[region] = (
            np.abs(self.reference[region] - self.canvas[region]).sum(axis=2)
            * self.weight_map[region]
        )

    # ------------------------------------------------------------------
    # Genetic algorithm

    def sample_positions(self, n, cell=8):
        """Sample n canvas positions with probability proportional to the
        current error, so strokes are tried where the painting is worst."""
        small = cv2.resize(
            self.error_map,
            (max(1, self.w // cell), max(1, self.h // cell)),
            interpolation=cv2.INTER_AREA,
        )
        p = small.flatten().astype(np.float64)
        if p.sum() <= 0:
            p = np.ones_like(p)
        idx = self.rng.choice(len(p), size=n, p=p / p.sum())
        sh, sw = small.shape
        ys, xs = idx // sw, idx % sw
        cx = (xs + self.rng.random(n)) * (self.w / sw)
        cy = (ys + self.rng.random(n)) * (self.h / sh)
        return cx, cy

    def random_population(self, n, size_lo, size_hi):
        cx, cy = self.sample_positions(n)
        population = []
        for i in range(n):
            # Most strokes start aligned with the local edge direction;
            # some stay fully random so evolution keeps its diversity.
            px = int(np.clip(cx[i], 0, self.w - 1))
            py = int(np.clip(cy[i], 0, self.h - 1))
            if self.oriented[py, px] and self.rng.random() < 0.7:
                angle = (self.orientation[py, px] + self.rng.normal(0, 15)) % 360
            else:
                angle = self.rng.uniform(0, 360)
            population.append(
                {
                    "brush": random.randrange(len(self.brushes)),
                    "cx": cx[i],
                    "cy": cy[i],
                    "size": self.rng.uniform(size_lo, size_hi),
                    "angle": angle,
                    "opacity": self.rng.uniform(0.6, 1.0),
                    "center_bias": self.rng.uniform(0.0, 0.5),
                    "jitter": np.zeros(3, np.float32),
                }
            )
        return population

    def mutate(self, stroke, size_lo, size_hi):
        child = dict(stroke)
        child["jitter"] = stroke["jitter"].copy()
        child["cx"] = float(
            np.clip(stroke["cx"] + self.rng.normal(0, stroke["size"] * 0.25), 0, self.w - 1)
        )
        child["cy"] = float(
            np.clip(stroke["cy"] + self.rng.normal(0, stroke["size"] * 0.25), 0, self.h - 1)
        )
        child["size"] = float(
            np.clip(stroke["size"] * self.rng.uniform(0.8, 1.25), size_lo, size_hi)
        )
        child["angle"] = (stroke["angle"] + self.rng.uniform(-30, 30)) % 360
        child["opacity"] = float(np.clip(stroke["opacity"] + self.rng.uniform(-0.1, 0.1), 0.3, 1.0))
        child["center_bias"] = float(
            np.clip(stroke["center_bias"] + self.rng.uniform(-0.2, 0.2), 0.0, 1.0)
        )
        child["jitter"] = np.clip(
            child["jitter"] + self.rng.uniform(-10, 10, 3).astype(np.float32), -40, 40
        )
        if self.rng.random() < 0.1:
            child["brush"] = random.randrange(len(self.brushes))
        return child

    def group_fitness(self, group):
        """Approximate fitness of a stroke group: sum of each stroke's
        delta against the current canvas, ignoring stroke-stroke overlap
        (positions come from error-map sampling, so overlap is rare).
        Exact re-evaluation happens at commit time. Invalid (off-canvas)
        strokes get a penalty large enough to always lose to any group
        of real strokes."""
        total = 0.0
        for stroke in group:
            d = self.stroke_delta(stroke)
            total += d if np.isfinite(d) else 1e9
        return total

    def evolve_group(self, group_size, pop_size, generations, size_lo, size_hi,
                     elite_frac=0.25):
        """Evolve a group of strokes jointly (anopara-style DNA with
        uniform crossover). group_size 1 degenerates to a single-stroke
        GA with mutation only."""
        flat = self.random_population(group_size * pop_size, size_lo, size_hi)
        population = [
            flat[i * group_size : (i + 1) * group_size] for i in range(pop_size)
        ]
        scored = sorted(
            ((self.group_fitness(g), g) for g in population), key=lambda t: t[0]
        )
        n_elite = max(2, int(pop_size * elite_frac))
        for _ in range(generations - 1):
            elites = scored[:n_elite]
            children = []
            for _ in range(pop_size - n_elite):
                pa = random.choice(elites)[1]
                pb = random.choice(elites)[1]
                child = []
                mutated = False
                for k in range(group_size):
                    gene = pa[k] if self.rng.random() < 0.5 else pb[k]
                    if group_size == 1 or self.rng.random() < 0.5:
                        gene = self.mutate(gene, size_lo, size_hi)
                        mutated = True
                    else:
                        gene = dict(gene)
                        gene["jitter"] = gene["jitter"].copy()
                    child.append(gene)
                if not mutated:
                    k = random.randrange(group_size)
                    child[k] = self.mutate(child[k], size_lo, size_hi)
                children.append((self.group_fitness(child), child))
            scored = sorted(elites + children, key=lambda t: t[0])
        return scored[0]

    def generate(
        self,
        strokes=1500,
        population_size=32,
        generations=10,
        min_size=None,
        max_size=None,
        group=1,
        out_dir="output",
        save_every=100,
    ):
        os.makedirs(out_dir, exist_ok=True)
        short = min(self.h, self.w)
        if max_size is None:
            max_size = short * 0.5
        if min_size is None:
            min_size = max(6, short * 0.015)

        painted = 0
        for i in range(strokes):
            # Anneal stroke size geometrically from max_size down to min_size.
            t = i / max(1, strokes - 1)
            center = max_size * (min_size / max_size) ** t
            size_lo, size_hi = center * 0.6, center * 1.4

            _, best_group = self.evolve_group(
                max(1, group), population_size, generations, size_lo, size_hi
            )
            # Group fitness ignores stroke-stroke overlap, so re-check each
            # stroke exactly against the evolving canvas: the painting
            # never gets worse.
            for stroke in best_group:
                if self.stroke_delta(stroke) < 0:
                    self.commit(stroke)
                    painted += 1

            if (i + 1) % save_every == 0 or i == strokes - 1:
                err = self.error_map.mean()
                unit = "round" if group > 1 else "stroke"
                print(
                    f"{unit} {i + 1}/{strokes}  painted={painted}  "
                    f"mean error/px={err:.2f}",
                    flush=True,
                )
                cv2.imwrite(
                    os.path.join(out_dir, f"progress_{i + 1:05d}.jpg"),
                    self.canvas.astype(np.uint8),
                )

        final = os.path.join(out_dir, "final.png")
        cv2.imwrite(final, self.canvas.astype(np.uint8))
        print(f"done: {final}")
        return self.canvas.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Paint an image with an evolved stroke sequence.")
    parser.add_argument("image", nargs="?", default="test.jpg", help="reference image")
    parser.add_argument("--shapes", default="shapes/*.png",
                        help="brush images: a glob or a directory of PNGs "
                             "(alpha channel used if present, else luminance)")
    parser.add_argument("--group", type=int, default=1,
                        help="strokes evolved jointly per round (1 = greedy)")
    parser.add_argument("--strokes", type=int, default=1500, help="number of strokes to paint")
    parser.add_argument("--population", type=int, default=32, help="GA population per stroke")
    parser.add_argument("--generations", type=int, default=10, help="GA generations per stroke")
    parser.add_argument("--max-dim", type=int, default=1200, help="working resolution (long side)")
    parser.add_argument("--canvas", choices=["mean", "black"], default="mean",
                        help="starting canvas color")
    parser.add_argument("--init-canvas", default=None,
                        help="resume from a previous snapshot image")
    parser.add_argument("--edge-weight", type=float, default=1.0,
                        help="extra fitness weight on reference edges (0 = plain L1)")
    parser.add_argument("--min-size", type=float, default=None,
                        help="smallest stroke size in px (default: 1.5%% of short side)")
    parser.add_argument("--max-size", type=float, default=None,
                        help="largest stroke size in px (default: 50%% of short side)")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--save-every", type=int, default=100, help="snapshot interval (strokes)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if os.path.isdir(args.shapes):
        shapes = sorted(
            os.path.join(args.shapes, name)
            for name in os.listdir(args.shapes)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    else:
        shapes = sorted(glob.glob(args.shapes))
    gp = GeneticPainting(args.image, shapes, seed=args.seed, max_dim=args.max_dim,
                         canvas=args.canvas, init_canvas=args.init_canvas,
                         edge_weight=args.edge_weight)
    gp.generate(
        strokes=args.strokes,
        population_size=args.population,
        generations=args.generations,
        min_size=args.min_size,
        max_size=args.max_size,
        group=args.group,
        out_dir=args.out,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
