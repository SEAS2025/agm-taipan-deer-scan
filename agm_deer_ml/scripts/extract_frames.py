"""
Extract frames from thermal video files or RTSP for labeling / training.

Examples:
  python extract_frames.py --video deer_clip.mp4 --out ../custom_frames --every 5
  python extract_frames.py --rtsp "rtsp://admin:abcd1234@10.15.12.1:554/live" --out ../custom_frames --seconds 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

DEFAULT_RTSP = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"


def extract_video(path: Path, out: Path, every: int, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    out.mkdir(parents=True, exist_ok=True)
    n = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            fn = out / f"{path.stem}_f{n:06d}.jpg"
            cv2.imwrite(str(fn), frame)
            saved += 1
        n += 1
    cap.release()
    print(f"Saved {saved} frames from {path.name} -> {out}")


def extract_rtsp(url: str, out: Path, seconds: float, every: int):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open RTSP: {url}")
    out.mkdir(parents=True, exist_ok=True)
    end = time.time() + seconds
    n = saved = 0
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if n % every == 0:
            fn = out / f"rtsp_{int(time.time())}_{n:06d}.jpg"
            cv2.imwrite(str(fn), frame)
            saved += 1
        n += 1
    cap.release()
    print(f"Saved {saved} RTSP frames -> {out}")
    print("Label with Roboflow Label Studio, or create .txt YOLO boxes (class 0 = deer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="local video file")
    ap.add_argument("--rtsp", help="RTSP URL")
    ap.add_argument("--out", type=Path, default=Path("../custom_frames"))
    ap.add_argument("--every", type=int, default=10, help="save every Nth frame")
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--seconds", type=float, default=30, help="RTSP capture duration")
    args = ap.parse_args()

    out = args.out.resolve()
    if args.video:
        extract_video(args.video.resolve(), out, args.every, args.max_frames)
    elif args.rtsp:
        extract_rtsp(args.rtsp, out, args.seconds, args.every)
    else:
        ap.print_help()
        print("\nProvide --video or --rtsp")


if __name__ == "__main__":
    main()
