"""
Thermal deer detection for AGM Taipan (ambulance-mounted roadside scan).

Usage (connect to Taipan hotspot first):
  python agm_deer_scanner.py
  python agm_deer_scanner.py --model agm_deer_ml/models/deer_thermal_best.pt
  python agm_deer_scanner.py --calibrate   # tune sensitivity on empty scene
  python agm_deer_scanner.py --no-audio    # visual only

Keys: q/ESC quit | s snapshot | r reconnect | +/- sensitivity | m mute
Use the Sensitivity slider window to tune deer detection.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

DEFAULT_URL = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"

DEER_PROFILE = {
    "name": "Cervidae (deer)",
    "core_temp_c": 38.5,
    "min_hot_delta": 10,
    "min_area_px": 120,
    "max_area_px": 22000,
    "min_aspect": 0.25,
    "max_aspect": 4.5,
    "min_extent": 0.22,
    "min_solidity": 0.45,
    "edge_core_ratio_min": 1.04,
    "confirm_frames": 3,
    "score_threshold": 0.52,
}


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    label: str
    centroid: tuple[int, int]


@dataclass
class DetectorState:
    bg: Optional[np.ndarray] = None
    sensitivity: float = 1.0
    track_history: dict = field(default_factory=dict)
    next_track_id: int = 0


class ThermalDeerDetector:
    """Hot-blob detector with deer-shaped thermal signature scoring."""

    def __init__(self, profile: dict | None = None):
        self.p = profile or DEER_PROFILE
        self.state = DetectorState()

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _update_background(self, gray: np.ndarray):
        g = gray.astype(np.float32)
        if self.state.bg is None:
            self.state.bg = g.copy()
            return
        alpha = 0.02 * self.state.sensitivity
        self.state.bg = (1 - alpha) * self.state.bg + alpha * g

    def _thermal_signature_score(self, gray: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() < 50:
            return 0.0
        core = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
        ring = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
        ring = cv2.subtract(ring, mask)
        if core.sum() < 10 or ring.sum() < 10:
            mean_in = float(gray[mask > 0].mean())
            mean_bg = float(self.state.bg[mask == 0].mean()) if (mask == 0).any() else 0
            return min(1.0, max(0, (mean_in - mean_bg) / 80.0))

        mean_core = float(gray[core > 0].mean())
        mean_ring = float(gray[ring > 0].mean())
        mean_bg = float(self.state.bg[mask == 0].mean()) if (mask == 0).any() else mean_ring

        edge_core = mean_core / max(mean_ring, 1)
        contrast = (mean_core - mean_bg) / max(self.p["min_hot_delta"], 1)
        ec_ok = edge_core >= self.p["edge_core_ratio_min"]
        score = 0.45 * min(1.0, contrast) + 0.35 * (1.0 if ec_ok else 0.3)
        score += 0.2 * min(1.0, edge_core - 1.0)
        return float(np.clip(score, 0, 1))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = self._to_gray(frame)
        self._update_background(gray)

        local_mean = cv2.GaussianBlur(gray, (51, 51), 0)
        local_diff = gray.astype(np.int16) - local_mean.astype(np.int16)

        delta = int(self.p["min_hot_delta"] * self.state.sensitivity)
        hot_local = (local_diff > delta).astype(np.uint8) * 255

        bg = self.state.bg.astype(np.uint8)
        global_diff = cv2.subtract(gray, bg)
        hot_global = (global_diff > delta).astype(np.uint8) * 255
        hot_pct = (gray > np.percentile(gray, 92)).astype(np.uint8) * 255

        mask = cv2.bitwise_or(hot_local, cv2.bitwise_or(hot_global, hot_pct))
        mask[:48, :] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p["min_area_px"] or area > self.p["max_area_px"]:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)
            if aspect < self.p["min_aspect"] or aspect > self.p["max_aspect"]:
                continue
            extent = area / max(w * h, 1)
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if extent < self.p["min_extent"] or solidity < self.p["min_solidity"]:
                continue

            blob_mask = np.zeros_like(gray)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            sig = self._thermal_signature_score(gray, blob_mask)

            shape_score = 1.0 if 0.4 <= aspect <= 2.8 else 0.6
            size_score = 1.0 if self.p["min_area_px"] * 2 <= area <= self.p["max_area_px"] * 0.5 else 0.7
            score = 0.55 * sig + 0.25 * shape_score + 0.20 * size_score

            if score >= self.p["score_threshold"] * 0.85:
                cx, cy = x + w // 2, y + h // 2
                dets.append(
                    Detection(
                        bbox=(x, y, w, h),
                        score=score,
                        label="DEER?" if score >= self.p["score_threshold"] else "heat",
                        centroid=(cx, cy),
                    )
                )

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:8]


class YoloDeerDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.35):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.state = DetectorState()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False, imgsz=640)
        dets: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                score = float(box.conf[0])
                cls = int(box.cls[0]) if box.cls is not None else 0
                name = r.names.get(cls, "deer") if r.names else "deer"
                dets.append(
                    Detection(
                        bbox=(x, y, w, h),
                        score=score,
                        label=name.upper(),
                        centroid=(x + w // 2, y + h // 2),
                    )
                )
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets


def make_detector(model: Optional[str]) -> Union[ThermalDeerDetector, YoloDeerDetector]:
    if model:
        p = Path(model)
        if not p.exists():
            raise SystemExit(f"Model not found: {p}")
        print(f"Using YOLO model: {p}")
        return YoloDeerDetector(p)
    default = Path(__file__).resolve().parent / "agm_deer_ml" / "models" / "deer_thermal_best.pt"
    if default.exists():
        print(f"Using trained YOLO model: {default}")
        return YoloDeerDetector(default)
    print("Using thermal heuristic detector (train YOLO: agm_deer_ml/scripts/train_deer_yolo.py)")
    return ThermalDeerDetector()


class AlertManager:
    """Speaker alerts with cooldown — single TTS worker thread (COM-safe on Windows)."""

    def __init__(self, cooldown: float = 8.0, enabled: bool = True):
        self.cooldown = cooldown
        self.enabled = enabled
        self.muted = False
        self._last_alert = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self._worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._worker.start()

    def _run(self):
        com_ready = False
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
            com_ready = True
        except Exception:
            pass

        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception as e:
            print(f"TTS init failed ({e}); using beep only")

        try:
            while True:
                text = self._queue.get()
                if text is None:
                    break
                spoke = False
                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                        spoke = True
                    except Exception:
                        pass
                if not spoke:
                    try:
                        import winsound
                        for _ in range(3):
                            winsound.Beep(880, 200)
                            time.sleep(0.08)
                    except Exception:
                        print("\aDEER DETECTED!\a")
        finally:
            if com_ready:
                try:
                    import pythoncom  # type: ignore
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def trigger(self, count: int = 1):
        if not self.enabled or self.muted:
            return
        now = time.time()
        with self._lock:
            if now - self._last_alert < self.cooldown:
                return
            self._last_alert = now
        msg = "Deer detected ahead" if count == 1 else f"{count} deer detected ahead"
        print(f"\n*** ALERT: {msg} ***")
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            pass


def open_capture(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


SENS_WINDOW = "Deer Scan — Sensitivity"


def _sens_to_trackbar(use_yolo: bool, value: float) -> int:
    if use_yolo:
        tb = int((value - 0.15) / 0.70 * 100)
    else:
        tb = int((value - 0.5) / 1.5 * 100)
    return max(0, min(100, tb))


def _trackbar_to_sens(use_yolo: bool, tb: int) -> float:
    if use_yolo:
        return 0.15 + (tb / 100.0) * 0.70
    return 0.5 + (tb / 100.0) * 1.5


def setup_sensitivity_control(use_yolo: bool, initial: float):
    cv2.namedWindow(SENS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SENS_WINDOW, 360, 70)
    cv2.createTrackbar(
        "Sensitivity",
        SENS_WINDOW,
        _sens_to_trackbar(use_yolo, initial),
        100,
        lambda _v: None,
    )


def read_sensitivity(use_yolo: bool, detector) -> float:
    try:
        tb = cv2.getTrackbarPos("Sensitivity", SENS_WINDOW)
    except cv2.error:
        return detector.conf if use_yolo else detector.state.sensitivity
    value = _trackbar_to_sens(use_yolo, tb)
    if use_yolo:
        detector.conf = value
    else:
        detector.state.sensitivity = value
    return value


def set_sensitivity_trackbar(use_yolo: bool, value: float):
    try:
        cv2.setTrackbarPos("Sensitivity", SENS_WINDOW, _sens_to_trackbar(use_yolo, value))
    except cv2.error:
        pass


def draw_hud(
    frame: np.ndarray,
    dets: list[Detection],
    status: str,
    fps: float,
    armed: bool,
    sensitivity: float,
):
    out = frame.copy()
    h, w = out.shape[:2]

    for d in dets:
        x, y, bw, bh = d.bbox
        is_deer = d.score >= DEER_PROFILE["score_threshold"]
        color = (0, 0, 255) if is_deer else (0, 165, 255)
        thickness = 3 if is_deer else 1
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        label = f"{d.label} {d.score:.0%}"
        cv2.putText(out, label, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

    bar_color = (0, 0, 255) if armed else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"DEER SCAN | {status} | {fps:.1f} FPS | sens {sensitivity:.1f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        bar_color,
        2,
    )
    cv2.putText(
        out,
        "q quit | s snap | r reconnect | +/- sens | m mute | use Sensitivity slider",
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="AGM Taipan live deer thermal scanner")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--cooldown", type=float, default=8.0, help="seconds between voice alerts")
    ap.add_argument("--calibrate", action="store_true", help="10s background learn, then scan")
    ap.add_argument("--model", help="path to YOLO .pt weights")
    ap.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    args = ap.parse_args()

    detector = make_detector(args.model)
    use_yolo = isinstance(detector, YoloDeerDetector)
    if use_yolo:
        detector.conf = args.yolo_conf
    alerts = AlertManager(cooldown=args.cooldown, enabled=not args.no_audio)

    print("=" * 60)
    print("AGM Taipan — Live Deer Thermal Scanner")
    print("=" * 60)
    print(f"Stream: {args.url}")
    mode = "YOLO" if use_yolo else f"Thermal heuristic ({DEER_PROFILE['name']})"
    print(f"Detector: {mode}")
    if not use_yolo:
        print(f"Core temp ref ~{DEER_PROFILE['core_temp_c']} C")
    print("Mount: forward-facing on ambulance. Best at night / cool ambient.")
    print("Keys: q quit | s snapshot | r reconnect | +/- sensitivity | m mute")
    print("Sensitivity slider: separate small window (lower = fewer alerts)\n")

    cap = None
    window = "Deer Scan — AGM Thermal"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    initial_sens = detector.conf if use_yolo else detector.state.sensitivity
    setup_sensitivity_control(use_yolo, initial_sens)

    confirm_streak = 0
    last_ok = time.time()
    fps_t = time.time()
    fps_n = 0
    fps = 0.0
    calib_until = time.time() + 10 if args.calibrate else 0
    score_threshold = DEER_PROFILE["score_threshold"]

    while True:
        if cap is None or not cap.isOpened():
            print(f"[{time.strftime('%H:%M:%S')}] Connecting ...")
            if cap:
                cap.release()
            cap = open_capture(args.url)
            if not cap.isOpened():
                time.sleep(2)
                continue
            print("Connected.")
            calib_until = time.time() + 10 if args.calibrate else 0

        ok, frame = cap.read()
        if not ok or frame is None:
            if time.time() - last_ok > 3:
                cap.release()
                cap = None
            cv2.waitKey(1)
            continue
        last_ok = time.time()

        fps_n += 1
        if time.time() - fps_t >= 1:
            fps = fps_n / (time.time() - fps_t)
            fps_n = 0
            fps_t = time.time()

        sens = read_sensitivity(use_yolo, detector)

        if calib_until > time.time() and not use_yolo:
            detector._update_background(detector._to_gray(frame))
            status = f"CALIBRATING {int(calib_until - time.time())}s"
            armed = False
            dets = []
        else:
            dets = detector.detect(frame)
            if use_yolo:
                deer_hits = [d for d in dets if d.score >= sens]
            else:
                eff_thresh = score_threshold / max(sens, 0.5)
                deer_hits = [d for d in dets if d.score >= eff_thresh]
            if deer_hits:
                confirm_streak += 1
            else:
                confirm_streak = max(0, confirm_streak - 1)

            armed = confirm_streak >= DEER_PROFILE["confirm_frames"]
            if armed:
                status = f"DEER ALERT ({len(deer_hits)})"
                alerts.trigger(len(deer_hits))
                confirm_streak = 0
            elif deer_hits:
                status = f"TRACKING ({len(deer_hits)})"
            else:
                status = "SCANNING"

        display = draw_hud(frame, dets, status, fps, armed, sens)
        if armed:
            cv2.putText(
                display, "!!! DEER !!!", (display.shape[1] // 2 - 120, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
            )
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            fn = f"deer_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, display)
            print(f"Saved {fn}")
        if key == ord("r"):
            cap.release()
            cap = None
        if key in (ord("+"), ord("=")):
            if use_yolo:
                detector.conf = min(0.85, detector.conf + 0.05)
                print(f"YOLO conf: {detector.conf:.2f}")
            else:
                detector.state.sensitivity = min(2.0, detector.state.sensitivity + 0.1)
                print(f"Sensitivity: {detector.state.sensitivity:.1f}")
            set_sensitivity_trackbar(use_yolo, detector.conf if use_yolo else detector.state.sensitivity)
        if key == ord("-"):
            if use_yolo:
                detector.conf = max(0.15, detector.conf - 0.05)
                print(f"YOLO conf: {detector.conf:.2f}")
            else:
                detector.state.sensitivity = max(0.5, detector.state.sensitivity - 0.1)
                print(f"Sensitivity: {detector.state.sensitivity:.1f}")
            set_sensitivity_trackbar(use_yolo, detector.conf if use_yolo else detector.state.sensitivity)
        if key == ord("m"):
            alerts.muted = not alerts.muted
            print("Audio", "MUTED" if alerts.muted else "ON")

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Scanner stopped.")


if __name__ == "__main__":
    main()
