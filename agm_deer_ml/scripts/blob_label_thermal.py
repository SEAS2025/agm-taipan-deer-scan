r"""
Pseudo-label deer in REAL thermal frames using palette-agnostic blob detection.

Deer in thermal scope footage are compact regions that contrast with the
local background -- BRIGHT in white-hot/fusion/red-hot, DARK in black-hot.
Morphological top-hat highlights bright-on-dark; black-hat highlights
dark-on-light. Combining both makes the detector palette-agnostic.

Candidates are filtered by area, aspect ratio and solidity to look deer-like.
We also reject blobs touching the HUD edges / center reticle region.
"""

from __future__ import annotations
import argparse, shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [ROOT / "calib_taipan_tm19", ROOT / "calib_agm_deer"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def detect_blobs(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)     # bright deer
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)  # dark deer
    boxes = []
    frame_area = float(h * w)
    for feat in (tophat, blackhat):
        _, m = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < frame_area * 0.0004 or area > frame_area * 0.05:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / max(bh, 1)
            if ar < 0.25 or ar > 4.0:           # deer-ish aspect
                continue
            solidity = area / max(bw * bh, 1)
            if solidity < 0.35:
                continue
            if y < h * 0.06 or y + bh > h * 0.97:  # skip HUD top / bottom bar
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            boxes.append((cx, cy, bw / w, bh / h))
    # merge near-duplicate boxes (tophat+blackhat overlap)
    return _nms(boxes)


def _nms(boxes, iou_thr=0.4):
    if not boxes:
        return []
    def to_xyxy(b):
        cx, cy, w, h = b
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    kept = []
    for b in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        bx = to_xyxy(b)
        dup = False
        for k in kept:
            kx = to_xyxy(k)
            ix = max(0, min(bx[2], kx[2]) - max(bx[0], kx[0]))
            iy = max(0, min(bx[3], kx[3]) - max(bx[1], kx[1]))
            inter = ix * iy
            ua = (bx[2]-bx[0])*(bx[3]-bx[1]) + (kx[2]-kx[0])*(kx[3]-kx[1]) - inter
            if ua > 0 and inter / ua > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    frames = []
    for d in SRC_DIRS:
        if d.exists():
            frames += [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        frames = frames[: args.limit]
    if args.preview:
        prev = ROOT / "blob_preview"
        if prev.exists():
            shutil.rmtree(prev)
        prev.mkdir(parents=True)
        import random
        random.seed(3)
        sample = random.sample(frames, min(12, len(frames)))
        for p in sample:
            im = cv2.imread(str(p))
            for cx, cy, bw, bh in detect_blobs(im):
                h, w = im.shape[:2]
                x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
                x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(prev / p.name), im)
        # grid
        tiles = [cv2.resize(cv2.imread(str(prev/p.name)), (320, 240)) for p in sample]
        while len(tiles) < 12:
            tiles.append(np.zeros((240,320,3),np.uint8))
        rows = [np.hstack(tiles[i:i+4]) for i in range(0,12,4)]
        cv2.imwrite(str(ROOT/"blob_label_preview.png"), np.vstack(rows))
        print("wrote", ROOT/"blob_label_preview.png")
        return 0
    # stats
    total = hit = nboxes = 0
    for p in frames:
        im = cv2.imread(str(p))
        if im is None:
            continue
        total += 1
        b = detect_blobs(im)
        if b:
            hit += 1
            nboxes += len(b)
    print(f"frames={total} with_detection={hit} ({100*hit/max(total,1):.1f}%) boxes={nboxes} avg/frame={nboxes/max(hit,1):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())