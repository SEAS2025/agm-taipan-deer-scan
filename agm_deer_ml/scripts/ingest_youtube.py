"""
Ingest a YouTube thermal hunting video into the training dataset.

Downloads the video with yt-dlp, samples frames at a fixed interval, and
writes them into dataset_internet/thermal/ so the existing
auto-label -> prepare -> train pipeline picks them up automatically.

Examples:
  python ingest_youtube.py --url https://www.youtube.com/watch?v=Ag2MsU9tl_o
  python ingest_youtube.py --url <URL> --every 20 --max-frames 600 --start 120 --end 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
THERMAL_DIR = ROOT / "dataset_internet" / "thermal"
DOWNLOAD_DIR = ROOT / "dataset_internet" / "_youtube"


def download_video(url: str, dst_dir: Path) -> Path:
    import yt_dlp

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Progressive single-file formats only (video+audio muxed) so no ffmpeg
    # merge step is required. 22 = 720p mp4, 18 = 360p mp4.
    opts = {
        "format": "22/18/best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none]",
        "outtmpl": str(dst_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid = info.get("id", "video")
        title = info.get("title", "")
    print(f"Downloaded: {title} [{vid}]")
    for ext in ("mp4", "mkv", "webm"):
        cand = dst_dir / f"{vid}.{ext}"
        if cand.exists():
            return cand
    matches = sorted(dst_dir.glob(f"{vid}.*"))
    if matches:
        return matches[0]
    raise SystemExit("Download finished but no video file was found.")


def sample_frames(
    video: Path,
    out: Path,
    every: int,
    max_frames: int,
    start_s: float,
    end_s: float | None,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out.mkdir(parents=True, exist_ok=True)

    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    stem = video.stem
    n = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end_s is not None and t > end_s:
            break
        if n % every == 0:
            fn = out / f"yt_{stem}_f{n:06d}.jpg"
            cv2.imwrite(str(fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        n += 1
    cap.release()
    print(f"Sampled {saved} frames -> {out}")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a YouTube thermal video into the deer dataset")
    ap.add_argument("--url", required=True, help="YouTube video URL")
    ap.add_argument("--out", type=Path, default=THERMAL_DIR, help="output frame folder")
    ap.add_argument("--every", type=int, default=25, help="save every Nth decoded frame")
    ap.add_argument("--max-frames", type=int, default=500, help="max frames to keep")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--keep-video", action="store_true", help="keep the downloaded video file")
    args = ap.parse_args()

    print(f"Fetching {args.url}")
    video = download_video(args.url, DOWNLOAD_DIR)
    saved = sample_frames(video, args.out.resolve(), args.every, args.max_frames, args.start, args.end)

    if not args.keep_video:
        try:
            video.unlink()
        except OSError:
            pass

    print(
        f"\nDone. {saved} thermal frames added to {args.out}\n"
        "Next: auto-label + train\n"
        "  python agm_deer_ml/scripts/auto_label_deer.py\n"
        "  python agm_deer_ml/scripts/prepare_dataset.py --clean --merge-internet\n"
        "  python agm_deer_ml/scripts/train_deer_yolo.py --epochs 30 --batch 4\n"
        "Or just click Start training in the web app."
    )
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())