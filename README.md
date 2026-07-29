# GeneticPainting

Procedural painting in the spirit of [genetic-drawing](https://github.com/anopara/genetic-drawing):
a reference image is approximated by layering brush strokes, where each stroke
is found with a small genetic algorithm.

![final painting](output/final.png)

## How it works

1. The canvas starts as the reference's mean color.
2. For each stroke, a population of random candidates (brush shape, position,
   size, rotation, opacity, color jitter) is evolved for a few generations.
   Fitness is the change in per-pixel error vs. the reference, evaluated only
   inside the stroke's bounding box.
3. Candidate positions are sampled from the current error map, so strokes are
   tried where the painting is still worst.
4. The winning stroke is alpha-composited onto the canvas if it improves the
   error. Stroke sizes anneal from coarse to fine, so early strokes block in
   large color regions and later strokes add detail.

Stroke colors are taken from the mean reference color under the brush
footprint, blended toward the color at the stroke center by an evolved
"center bias" gene (so small bright features like a moon or lit windows
aren't averaged away), plus an evolved jitter.

Fitness is edge-weighted (`--edge-weight`): per-pixel error is scaled up
near edges of the reference, because plain L1 error underweights small,
crisp features — a pale moon on pale clouds is numerically almost
invisible but perceptually obvious.

## Usage

```bash
pip install -r requirements.txt

# regenerate the random brush masks (optional, 10 already included)
python shapes.py

# paint (defaults: test.jpg, 1500 strokes)
python generate.py test.jpg --strokes 3000 --max-dim 1600 --out output

# optional: refine an existing result with small strokes and blob brushes only
python generate.py test.jpg --shapes 'shapes/[2346789].png' \
    --init-canvas output/final.png --strokes 700 --min-size 7 --max-size 70 \
    --max-dim 1600 --edge-weight 3 --out output
```

Progress snapshots are written to `output/progress_*.jpg` and the result to
`output/final.png`. See `python generate.py --help` for all options
(population size, GA generations, working resolution, starting canvas,
edge weighting, resume canvas, seed).
