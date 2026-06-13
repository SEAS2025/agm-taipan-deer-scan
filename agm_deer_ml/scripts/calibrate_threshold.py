r"""
Calibrate the deer-detection confidence threshold for SAFE (high-recall) use.

Runs the trained model over real thermal frames (ideally captured from the
Taipan scope) and reports, for each confidence threshold, the fraction of
deer frames that fire at least one detection (recall proxy) plus the average
false detections on background frames.

Usage:
  # Frames that DO contain a deer (recall):
  python calibrate_threshold.py --deer-dir path\to\deer_frames
  # Optionally add background frames (no deer) to estimate false alarms:
  python calibrate_threshold.py --deer-dir deer_frames --bg-dir empty_frames
  # From a video clip of deer:
  python calibrate_threshold.py --deer-video clip.mp4 --every 5

  --target-recall 0.95   target detection rate to pick the threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "deer_thermal_best.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def frames_from_dir(d: Path):
    return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def frames_from_video(v: Path, every: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(v))
    n = saved = 0
    paths = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % every == 0:
            fp = out / f"{v.stem}_{n:06d}.jpg"
            cv2.imwrite(str(fp), fr)
            paths.append(fp)
            saved += 1
        n += 1
    cap.release()
    print(f"Extracted {saved} frames from {v.name}")
    return paths


def max_conf_per_frame(model, paths, conf_floor=0.03):
    """Return list of max detection confidence per frame (0 if none)."""
    out = []
    for p in paths:
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        res = model.predict(fr, conf=conf_floor, verbose=False)
        best = 0.0
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                best = max(best, float(b.conf[0]))
        out.append(best)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate safe deer-detection threshold")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--deer-dir", type=Path, help="frames that contain a deer")
    ap.add_argument("--bg-dir", type=Path, help="frames with NO deer (optional)")
    ap.add_argument("--deer-video", type=Path, help="video clip of deer")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    deer_paths = []
    if args.deer_video:
        deer_paths = frames_from_video(args.deer_video, args.every, ROOT / "calib_frames")
    elif args.deer_dir:
        deer_paths = frames_from_dir(args.deer_dir)
    if not deer_paths:
        raise SystemExit("Provide --deer-dir or --deer-video with deer frames")

    print(f"Scoring {len(deer_paths)} deer frames...")
    deer_conf = max_conf_per_frame(model, deer_paths)

    bg_conf = []
    if args.bg_dir and args.bg_dir.exists():
        bg_paths = frames_from_dir(args.bg_dir)
        print(f"Scoring {len(bg_paths)} background frames...")
        bg_conf = max_conf_per_frame(model, bg_paths)

    print("\nthresh  recall(deer detected)  false-alarm(bg)")
    pick = None
    for t in THRESHOLDS:
        rec = sum(1 for c in deer_conf if c >= t) / max(1, len(deer_conf))
        fa = (sum(1 for c in bg_conf if c >= t) / len(bg_conf)) if bg_conf else float("nan")
        flag = ""
        if pick is None and rec >= args.target_recall:
            pick = t
            flag = "  <- safe threshold"
        fa_s = f"{fa:5.1%}" if bg_conf else "  n/a"
        print(f"{t:5.2f}   {rec:6.1%}                {fa_s}{flag}")

    print()
    if pick is not None:
        print(f"RECOMMENDED conf threshold for >= {args.target_recall:.0%} recall: {pick:.2f}")
        print(f"Set in scanner:  --conf {pick:.2f}   (or YOLO conf slider)")
    else:
        print(f"Could not reach {args.target_recall:.0%} recall even at {THRESHOLDS[0]:.2f}.")
        print("Model needs more/better deer training data for this footage.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())