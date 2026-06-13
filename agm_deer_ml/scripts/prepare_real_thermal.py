r"""
Build a REAL-thermal deer dataset from AGM/Taipan calibration frames.

Pipeline per frame:
  1. Normalize palette -> canonical grayscale with hot = bright
     (auto-inverts black-hot; luminance proxy works for white-hot/fusion/red-hot).
  2. Pseudo-label deer with YOLO-World on the normalized image (open-vocab),
     filtered by plausible box size/aspect.
  3. Emit the normalized image + an inverted copy (polarity invariance) with
     the same boxes, so the model handles white-hot AND black-hot.

Output:
  dataset_real/images/{train,val}, dataset_real/labels/{train,val}
A held-out fraction of SOURCE frames is reserved for honest calibration and
NOT used for training (written to calib_holdout/).
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [ROOT / "calib_taipan_tm19", ROOT / "calib_agm_deer"]
OUT = ROOT / "dataset_real"
HOLDOUT = ROOT / "calib_holdout"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEER_PROMPTS = ["deer", "animal", "mammal", "dog"]


def to_hot_bright_gray(bgr: np.ndarray) -> np.ndarray:
    """Return single-channel uint8 where hot objects are bright."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Otsu split; if the BRIGHT class covers most of the frame, it's likely
    # black-hot (subject dark on bright bg) -> invert so hot = bright.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_frac = float((mask > 0).mean())
    if bright_frac > 0.55:
        gray = 255 - gray
    return gray


def gray3(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def label_boxes(model, img3: np.ndarray, conf: float):
    h, w = img3.shape[:2]
    res = model.predict(img3, conf=conf, verbose=False)
    out = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw < 0.015 or bh < 0.02:      # too small = noise
                continue
            if bw > 0.6 or bh > 0.8:          # too big = whole-scene
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            out.append((cx, cy, bw, bh))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.06)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    if HOLDOUT.exists():
        shutil.rmtree(HOLDOUT)
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    HOLDOUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for d in SRC_DIRS:
        if d.exists():
            frames += [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]
    random.seed(7)
    random.shuffle(frames)

    n_hold = int(len(frames) * args.holdout_ratio)
    holdout = frames[:n_hold]
    trainable = frames[n_hold:]
    for p in holdout:
        shutil.copy2(p, HOLDOUT / p.name)

    from ultralytics import YOLO
    print("Loading YOLO-World for pseudo-labeling...")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(DEER_PROMPTS)

    kept = labeled = 0
    for i, p in enumerate(trainable):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = to_hot_bright_gray(bgr)
        boxes = label_boxes(model, gray3(gray), args.conf)
        if not boxes:
            continue  # only keep frames with a detected deer (positives)
        labeled += 1
        sp = "val" if (kept % int(1 / args.val_ratio) == 0) else "train"
        inv = 255 - gray
        for tag, im in (("n", gray), ("i", inv)):
            stem = f"real_{p.stem}_{tag}"
            cv2.imwrite(str(OUT / "images" / sp / f"{stem}.jpg"), im)
            (OUT / "labels" / sp / f"{stem}.txt").write_text(
                "\n".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes),
                encoding="utf-8",
            )
            kept += 1
        if labeled % 25 == 0:
            print(f"  {labeled} frames labeled ({kept} imgs incl. inverted)")

    n_tr = len(list((OUT / "images" / "train").glob("*")))
    n_va = len(list((OUT / "images" / "val").glob("*")))
    print(f"\nReal-thermal dataset: {n_tr} train, {n_va} val imgs from {labeled} labeled frames")
    print(f"Holdout (for calibration, untrained): {len(holdout)} frames -> {HOLDOUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())