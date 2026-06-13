r"""
Auto-label deer in STATIC (tripod) thermal footage via background subtraction.

Because the camera does not move, anything that moves is an animal. We learn the
static background (MOG2), extract moving foreground blobs, filter them to look
deer-like (size / aspect / solidity), and require temporal persistence to reject
flicker and sensor noise. This is palette-agnostic -- it keys on MOTION, not
appearance -- so white-hot, black-hot, fusion and red-hot all work.

Usage:
  # a clip that CONTAINS deer (auto-labels moving animals):
  python label_static_thermal.py --video feeder_whitehot.mp4 --palette whitehot
  # an EMPTY scene clip (no animals -> negative/background frames, no labels):
  python label_static_thermal.py --video empty_scene.mp4 --empty
  # whole folder of clips:
  python label_static_thermal.py --dir feeder_clips
  # sanity-check the boxes on a few frames without writing the dataset:
  python label_static_thermal.py --video feeder.mp4 --preview
"""

from __future__ import annotations
import argparse, shutil, random
from collections import deque
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset_feeder"
IMG_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def deerlike(area_frac, ar, solidity):
    return (0.0004 <= area_frac <= 0.08) and (0.3 <= ar <= 4.0) and (solidity >= 0.35)


def boxes_from_fgmask(fg, w, h):
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # drop MOG2 shadows (127)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    fa = float(w * h)
    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        solidity = area / max(bw * bh, 1)
        if deerlike(area / fa, ar, solidity):
            out.append((x, y, bw, bh))
    return out


def process_video(path: Path, empty: bool, every: int, preview: bool,
                  prev_tiles: list, writer):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  !! could not open {path.name}")
        return 0, 0
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24,
                                             detectShadows=True)
    idx = kept = 0
    palette = path.stem
    # require a box to persist across a short window to count as a real animal
    recent = deque(maxlen=3)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = mog.apply(gray)
        if idx < 30:                      # warm up background model
            continue
        boxes = [] if empty else boxes_from_fgmask(fg, w, h)
        recent.append(set((round(b[0]/w, 2), round(b[1]/h, 2)) for b in boxes))
        # keep frames: deer clips need >=1 persistent box; empty clips sample bg
        persistent = boxes and len(recent) == recent.maxlen and \
            all(len(r) > 0 for r in recent)
        if empty:
            take = (idx % every == 0)
        else:
            take = persistent and (idx % every == 0)
        if not take:
            continue
        if preview:
            im = frame.copy()
            for (x, y, bw, bh) in boxes:
                cv2.rectangle(im, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            if len(prev_tiles) < 12:
                prev_tiles.append(cv2.resize(im, (320, 240)))
            kept += 1
            continue
        writer(frame, boxes, f"{palette}_{path.stem}_{idx:06d}")
        kept += 1
    cap.release()
    print(f"  {path.name}: kept {kept} frames ({'empty/bg' if empty else 'deer'})")
    return kept, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--dir")
    ap.add_argument("--palette", default="")
    ap.add_argument("--empty", action="store_true")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="wipe dataset_feeder first")
    args = ap.parse_args()

    clips = []
    if args.video:
        clips.append(Path(args.video))
    if args.dir:
        d = Path(args.dir)
        clips += [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]
    if not clips:
        print("No clips given. Use --video FILE or --dir FOLDER.")
        return 2

    if args.preview:
        tiles: list = []
        for c in clips:
            process_video(c, args.empty, args.every, True, tiles, None)
            if len(tiles) >= 12:
                break
        while len(tiles) < 12:
            tiles.append(np.zeros((240, 320, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
        outp = ROOT / "feeder_label_preview.png"
        cv2.imwrite(str(outp), np.vstack(rows))
        print("wrote", outp)
        return 0

    if args.fresh and OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    random.seed(11)

    counter = {"n": 0}

    def writer(frame, boxes, stem):
        h, w = frame.shape[:2]
        sp = "val" if random.random() < args.val_ratio else "train"
        cv2.imwrite(str(OUT / "images" / sp / f"{stem}.jpg"), frame)
        lines = []
        for (x, y, bw, bh) in boxes:
            cx = (x + bw / 2) / w; cy = (y + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (OUT / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        counter["n"] += 1

    total = 0
    for c in clips:
        kept, _ = process_video(c, args.empty, args.every, False, [], writer)
        total += kept
    n_tr = len(list((OUT / "images" / "train").glob("*")))
    n_va = len(list((OUT / "images" / "val").glob("*")))
    print(f"\nFeeder dataset now: {n_tr} train / {n_va} val images.")
    print("Run with --preview first to eyeball box quality before training.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())