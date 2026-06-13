"""
Low-latency edge scanner for Raspberry Pi + AGM Taipan.

Optimized pipeline (no web UI):
  RTSP capture (drop-old-frames) -> YOLO inference -> distance -> tiered audio

Usage on Pi (connect to Taipan hotspot first):
  python agm_pi_scanner.py --model agm_deer_ml/models/deer_thermal_best.pt
  python agm_pi_scanner.py --model agm_deer_ml/models/deer_thermal_int8.tflite  # after export

Latency tips:
  --imgsz 416 --skip 1 --headless
  Export TFLite/NCNN on Pi: python agm_deer_ml/scripts/export_pi_model.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

from agm_deer_scanner import Detection
from agm_deer_alert import assess_nearest_threat
from agm_distance import DistanceEstimator
from agm_pi_audio import PiAudioAlerts

DEFAULT_URL = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"
ROOT = Path(__file__).resolve().parent


class LowLatencyCapture:
    """Always keep only the newest frame — drops stale frames for min latency."""

    def __init__(self, url: str):
        self.url = url
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._alive = True
        self.connected = False
        self._thread = threading.Thread(target=self._loop, name="rtsp-grab", daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        cap = None
        while self._alive:
            if cap is None or not cap.isOpened():
                self.connected = False
                if cap:
                    cap.release()
                cap = self._open()
                if not cap.isOpened():
                    time.sleep(1)
                    continue
                self.connected = True
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.05)
                continue
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((frame, time.time()))

    def read(self) -> tuple[bool, object | None, float]:
        try:
            frame, ts = self._q.get(timeout=2)
            return True, frame, ts
        except queue.Empty:
            return False, None, 0.0

    def stop(self):
        self._alive = False


def run_yolo(model_path: Path, frame, conf: float, imgsz: int) -> list[Detection]:
    from ultralytics import YOLO
    if not hasattr(run_yolo, "_model"):
        run_yolo._model = YOLO(str(model_path))
    results = run_yolo._model.predict(frame, conf=conf, verbose=False, imgsz=imgsz, half=False)
    dets: list[Detection] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            score = float(box.conf[0])
            dets.append(Detection(bbox=(x, y, w, h), score=score, label="DEER", centroid=(x + w // 2, y + h // 2)))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:6]


def main():
    ap = argparse.ArgumentParser(description="Pi edge deer scanner — low latency + distance audio")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=str(ROOT / "agm_deer_ml" / "models" / "deer_thermal_best.pt"))
    ap.add_argument("--conf", type=float, default=0.45, help="YOLO confidence (raise if model is 99% accurate)")
    ap.add_argument("--imgsz", type=int, default=416, help="inference size — 416 faster than 640 on Pi")
    ap.add_argument("--skip", type=int, default=0, help="run inference every N+1 frames (0=every frame)")
    ap.add_argument("--confirm", type=int, default=2, help="frames to confirm alert (lower = faster cue)")
    ap.add_argument("--zoom", type=float, default=1.0, help="digital zoom factor for distance calc")
    ap.add_argument("--headless", action="store_true", help="no display window (fastest on Pi)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-voice", action="store_true", help="buzzer only, no spoken callout")
    ap.add_argument("--calib", help="path to distance_calib.json")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first, then: python agm_deer_ml/scripts/export_pi_model.py")
        sys.exit(1)

    dist = DistanceEstimator.load(Path(args.calib) if args.calib else None)
    audio = PiAudioAlerts(enabled=not args.no_audio, voice=not args.no_voice)
    cap = LowLatencyCapture(args.url)

    print("=" * 60)
    print("AGM Pi Edge Scanner — EGPWS-style deer alerts")
    print("=" * 60)
    print(f"Model: {model_path.name} | imgsz={args.imgsz} | conf={args.conf}")
    print("Audio: EGPWS warning tone + Deer callout")
    print(f"Distance calib focal_px={dist.focal_px:.0f}")
    print("Ctrl+C to stop\n")

    confirm_streak = 0
    frame_i = 0
    last_dets: list[Detection] = []
    last_threat = None
    fps_t = time.time()
    fps_n = 0
    infer_ms = 0.0

    window = "Pi Deer Scan"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)

    try:
        while True:
            ok, frame, cap_ts = cap.read()
            if not ok or frame is None:
                continue

            frame_i += 1
            t0 = time.time()
            if frame_i % (args.skip + 1) == 0:
                last_dets = run_yolo(model_path, frame, args.conf, args.imgsz)
                infer_ms = (time.time() - t0) * 1000

            deer = [d for d in last_dets if d.score >= args.conf]
            fh, fw = frame.shape[:2]
            threat = assess_nearest_threat(deer, fw, dist, args.zoom) if deer else None
            last_threat = threat

            if deer:
                confirm_streak += 1
            else:
                confirm_streak = max(0, confirm_streak - 1)

            if confirm_streak >= args.confirm and threat:
                audio.trigger_threat(threat)
                confirm_streak = 0

            fps_n += 1
            if time.time() - fps_t >= 1:
                latency_ms = (time.time() - cap_ts) * 1000 if cap_ts else 0
                side = threat.side if threat else "—"
                dist_l = threat.distance.label if threat else "—"
                status = (
                    f"FPS {fps_n} | infer {infer_ms:.0f}ms | e2e ~{latency_ms:.0f}ms | "
                    f"deer {len(deer)} | {side} | {dist_l}"
                )
                print(status)
                fps_n = 0
                fps_t = time.time()

            if not args.headless:
                disp = frame.copy()
                for d in deer:
                    x, y, w, h = d.bbox
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    if threat and d is threat.detection:
                        label = f"{threat.side.upper()} {threat.distance.label}"
                        cv2.putText(disp, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow(window, disp)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Pi scanner stopped.")


if __name__ == "__main__":
    main()
