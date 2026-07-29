"""Paint a video with brushes — made for Bad Apple!! out of real apples.

The canvas persists across frames: after a frame is painted, the next
frame only differs where the video moved, so the error map concentrates
new strokes there (temporal coherence). Scene cuts get a bigger stroke
budget automatically because the frame-to-frame error is larger.

Usage (from the repo root):
    python examples/bad_apple.py bad_apple.mp4 --shapes examples/apples \
        --out bad_apple_painted.mp4
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import GeneticPainting, resolve_shapes  # noqa: E402


def paint_frame(gp, args):
    """Paint strokes until improvement plateaus or the budget runs out.
    The brush set has an irreducible error floor (real apples never match
    white perfectly), so stop on stalled progress rather than an absolute
    error target — static scenes cost one batch, scene cuts get hundreds
    of strokes. Returns the number of rounds used."""
    rounds = 0
    prev_err = float(gp.error_map.sum())
    while rounds < args.max_rounds:
        for _ in range(20):
            _, group = gp.evolve_group(
                1, args.population, args.generations, args.min_size, args.max_size
            )
            stroke = group[0]
            if gp.stroke_delta(stroke) < 0:
                gp.commit(stroke)
        rounds += 20
        err = float(gp.error_map.sum())
        if prev_err - err < args.tolerance * err:
            break
        prev_err = err
    return rounds


def mux_audio(tmp_video, args, out_idx):
    """Re-encode to H.264 and carry the source audio across."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_video, "-ss", str(args.start), "-i", args.video,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", args.out,
    ]
    subprocess.run(cmd, check=True)
    os.unlink(tmp_video)
    print(f"done: {args.out} ({out_idx} frames)")


def content_crop(frame):
    """Slice that trims constant-black pillar/letterbox borders."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.nonzero(gray.max(axis=0) > 8)[0]
    rows = np.nonzero(gray.max(axis=1) > 8)[0]
    if len(cols) == 0 or len(rows) == 0:
        return np.s_[:, :]
    return np.s_[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def paint_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    shapes = resolve_shapes(args.shapes)

    gp = None
    writer = None
    tmp_video = None
    crop = None
    out_idx = 0
    src_idx = 0
    start_frame = int(args.start * src_fps)
    end_frame = int((args.start + args.duration) * src_fps) if args.duration else None
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        src_idx = start_frame

    while True:
        ok, frame = cap.read()
        if not ok or (end_frame and src_idx >= end_frame):
            break
        # Emit at args.fps by picking the source frame whose timestamp
        # crosses each output tick.
        take = int((src_idx - start_frame) * args.fps / src_fps) >= out_idx
        src_idx += 1
        if not take:
            continue
        out_idx += 1

        if crop is None:
            crop = content_crop(frame)
        frame = frame[crop]

        h = int(round(frame.shape[0] * args.width / frame.shape[1] / 2) * 2)
        small = cv2.resize(frame, (args.width, h), interpolation=cv2.INTER_AREA)
        if args.lift:
            # Lift blacks to a tone dark brushes can actually reach, so
            # dark regions get tiled with dark apples instead of staying
            # flat canvas-black.
            small = (
                small.astype(np.float32) * ((255.0 - args.lift) / 255.0) + args.lift
            ).astype(np.uint8)

        if gp is None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                cv2.imwrite(tf.name, small)
                gp = GeneticPainting(
                    tf.name, shapes, seed=args.seed, max_dim=max(args.width, h),
                    canvas="black", edge_weight=args.edge_weight,
                    keep_brush_color=not args.paint,
                    brush_max_dim=int(2 * args.max_size),
                    sample_gamma=args.sample_gamma,
                )
            os.unlink(tf.name)
            tmp_video = args.out + ".noaudio.mp4"
            writer = cv2.VideoWriter(
                tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gp.w, gp.h)
            )
        else:
            gp.set_reference(small)

        rounds = paint_frame(gp, args)

        writer.write(gp.canvas.astype(np.uint8))
        if out_idx % 50 == 0:
            print(
                f"frame {out_idx}  (src {src_idx / src_fps:.1f}s)  "
                f"rounds={rounds}  err={gp.error_map.mean():.1f}",
                flush=True,
            )

    cap.release()
    if writer is None:
        raise ValueError("no frames read from video")
    writer.release()
    mux_audio(tmp_video, args, out_idx)


def main():
    parser = argparse.ArgumentParser(description="Paint a video with brushes.")
    parser.add_argument("video", help="source video")
    parser.add_argument("--shapes", default="examples/apples",
                        help="brush images: directory or glob")
    parser.add_argument("--out", default="painted.mp4")
    parser.add_argument("--width", type=int, default=480, help="output width")
    parser.add_argument("--fps", type=float, default=12.0, help="output fps")
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--duration", type=float, default=None, help="clip length (s)")
    parser.add_argument("--max-rounds", type=int, default=600,
                        help="stroke rounds cap per frame")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="stop when a 20-round batch improves total error "
                             "by less than this fraction")
    parser.add_argument("--sample-gamma", type=float, default=3.0,
                        help="stroke placement sharpness (error**gamma)")
    parser.add_argument("--lift", type=int, default=45,
                        help="lift video black level so dark brushes can tile "
                             "dark regions (0 = leave black flat)")
    parser.add_argument("--min-size", type=float, default=9)
    parser.add_argument("--max-size", type=float, default=55)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--edge-weight", type=float, default=2.0)
    parser.add_argument("--paint", action="store_true",
                        help="recolor brushes from the frame instead of "
                             "pasting their own pixels")
    parser.add_argument("--seed", type=int, default=0)
    paint_video(parser.parse_args())


if __name__ == "__main__":
    main()
