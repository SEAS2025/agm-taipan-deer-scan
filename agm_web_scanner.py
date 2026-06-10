"""
Lightweight local web scanner for AGM Taipan thermal deer detection.

  python agm_web_scanner.py
  python agm_web_scanner.py --demo          # synthetic feed, no scope needed
  python agm_web_scanner.py --port 8080

Open http://127.0.0.1:8080 in a browser (auto-opens on start).
Connect laptop to Taipan hotspot before starting (unless --demo).
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from agm_deer_scanner import (
    DEER_PROFILE,
    AlertManager,
    Detection,
    YoloDeerDetector,
    make_detector,
)
from agm_scope_control import ScopeControl

DEFAULT_URL = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"
WEB_DIR = Path(__file__).resolve().parent / "web"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")


@dataclass
class ScanState:
    url: str = DEFAULT_URL
    demo: bool = False
    sensitivity: float = 1.0
    yolo_conf: float = 0.35
    use_yolo: bool = False
    muted: bool = False
    status: str = "STARTING"
    fps: float = 0.0
    armed: bool = False
    detections: int = 0
    deer_hits: int = 0
    connected: bool = False
    reconnect_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    jpeg: Optional[bytes] = None
    jpeg_lock: threading.Lock = field(default_factory=threading.Lock)


state = ScanState()
detector = None
alerts: Optional[AlertManager] = None
scope: Optional[ScopeControl] = None


def draw_hud(
    frame: np.ndarray,
    dets: list[Detection],
    status: str,
    fps: float,
    armed: bool,
    sens: float,
):
    out = frame.copy()
    h, w = out.shape[:2]
    for d in dets:
        x, y, bw, bh = d.bbox
        is_deer = d.score >= DEER_PROFILE["score_threshold"]
        color = (0, 0, 255) if is_deer else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3 if is_deer else 1)
        cv2.putText(
            out,
            f"{d.label} {d.score:.0%}",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)
    bar_color = (0, 0, 255) if armed else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"DEER SCAN | {status} | {fps:.1f} FPS | sens {sens:.1f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        bar_color,
        2,
    )
    if armed:
        cv2.putText(
            out,
            "!!! DEER !!!",
            (w // 2 - 120, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
    return out


def open_capture(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class DemoFeed:
    """Synthetic thermal-like frames with moving hot blobs for UI testing."""

    def __init__(self, width: int = 640, height: int = 480):
        self.w = width
        self.h = height
        self.t = 0.0
        self.blobs = [
            {"x": 120, "y": 200, "vx": 0.8, "vy": 0.3, "r": 28},
            {"x": 400, "y": 280, "vx": -0.5, "vy": 0.2, "r": 22},
        ]

    def read(self) -> tuple[bool, np.ndarray]:
        self.t += 0.05
        rng = np.random.default_rng(int(self.t * 100) % 10000)
        gray = rng.integers(20, 45, (self.h, self.w), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        for b in self.blobs:
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            if b["x"] < 40 or b["x"] > self.w - 40:
                b["vx"] *= -1
            if b["y"] < 40 or b["y"] > self.h - 40:
                b["vy"] *= -1
            cx, cy, r = int(b["x"]), int(b["y"]), b["r"]
            cv2.circle(gray, (cx, cy), r, 220, -1)
            cv2.circle(gray, (cx, cy), max(4, r // 3), 255, -1)

        if math.sin(self.t * 0.7) > 0.85:
            ex = int(self.w * 0.65 + 30 * math.sin(self.t))
            ey = int(self.h * 0.35)
            cv2.ellipse(gray, (ex, ey), (18, 36), 0, 0, 360, 240, -1)

        frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        time.sleep(0.04)
        return True, frame


def process_frame(frame: np.ndarray, confirm_streak: int, score_threshold: float) -> tuple[np.ndarray, str, bool, int, int, int]:
    with state.lock:
        sens = state.yolo_conf if state.use_yolo else state.sensitivity

    dets = detector.detect(frame)
    if state.use_yolo:
        deer_hits = [d for d in dets if d.score >= sens]
    else:
        eff = score_threshold / max(sens, 0.5)
        deer_hits = [d for d in dets if d.score >= eff]

    if deer_hits:
        confirm_streak += 1
    else:
        confirm_streak = max(0, confirm_streak - 1)

    armed = confirm_streak >= DEER_PROFILE["confirm_frames"]
    if armed:
        status = f"DEER ALERT ({len(deer_hits)})"
        if alerts:
            with state.lock:
                alerts.muted = state.muted
            alerts.trigger(len(deer_hits))
        confirm_streak = 0
    elif deer_hits:
        status = f"TRACKING ({len(deer_hits)})"
    else:
        status = "SCANNING"

    display = draw_hud(frame, dets, status, 0.0, armed, sens)
    return display, status, armed, len(dets), len(deer_hits), confirm_streak


def capture_loop():
    global detector, alerts
    cap = None
    demo: Optional[DemoFeed] = None
    confirm_streak = 0
    fps_t = time.time()
    fps_n = 0
    fps = 0.0
    frame_i = 0
    score_threshold = DEER_PROFILE["score_threshold"]
    last_ok = time.time()

    while True:
        with state.lock:
            url = state.url
            demo_mode = state.demo
            if state.reconnect_requested:
                state.reconnect_requested = False
                if cap:
                    cap.release()
                    cap = None
                demo = None

        if demo_mode:
            if demo is None:
                demo = DemoFeed()
                with state.lock:
                    state.connected = True
                    state.status = "DEMO SCANNING"
            ok, frame = demo.read()
        else:
            if cap is None or not cap.isOpened():
                with state.lock:
                    state.status = "CONNECTING"
                    state.connected = False
                if cap:
                    cap.release()
                cap = open_capture(url)
                if not cap.isOpened():
                    time.sleep(2)
                    continue
                with state.lock:
                    state.connected = True
                    state.status = "SCANNING"
                confirm_streak = 0
                frame_i = 0

            ok, frame = cap.read()

        if not ok or frame is None:
            if not demo_mode and time.time() - last_ok > 3:
                if cap:
                    cap.release()
                    cap = None
            time.sleep(0.05)
            continue
        last_ok = time.time()
        frame_i += 1
        fps_n += 1
        if time.time() - fps_t >= 1:
            fps = fps_n / (time.time() - fps_t)
            fps_n = 0
            fps_t = time.time()

        if frame_i % 2 == 0:
            display, status, armed, det_n, deer_n, confirm_streak = process_frame(
                frame, confirm_streak, score_threshold
            )
            with state.lock:
                state.status = status
                state.fps = fps
                state.armed = armed
                state.detections = det_n
                state.deer_hits = deer_n
        else:
            with state.lock:
                status = state.status
                armed = state.armed
                sens = state.yolo_conf if state.use_yolo else state.sensitivity
            display = draw_hud(frame, [], status, fps, armed, sens)

        ok_enc, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok_enc:
            with state.jpeg_lock:
                state.jpeg = buf.tobytes()


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/video.mjpg")
def video():
    def gen():
        while True:
            with state.jpeg_lock:
                frame = state.jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state.lock:
        sens = state.yolo_conf if state.use_yolo else state.sensitivity
        scope_info = scope.as_dict() if scope else None
        return jsonify(
            status=state.status,
            fps=state.fps,
            armed=state.armed,
            detections=state.detections,
            deer_hits=state.deer_hits,
            connected=state.connected,
            sensitivity=sens,
            muted=state.muted,
            use_yolo=state.use_yolo,
            demo=state.demo,
            scope=scope_info,
        )


@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    data = request.get_json(force=True, silent=True) or {}
    val = float(data.get("value", 1.0))
    with state.lock:
        if state.use_yolo:
            state.yolo_conf = max(0.15, min(0.85, val))
            detector.conf = state.yolo_conf
        else:
            state.sensitivity = max(0.5, min(2.0, val))
            detector.state.sensitivity = state.sensitivity
    return jsonify(ok=True)


@app.route("/api/mute", methods=["POST"])
def api_mute():
    data = request.get_json(force=True, silent=True) or {}
    with state.lock:
        state.muted = bool(data.get("muted", False))
        if alerts:
            alerts.muted = state.muted
    return jsonify(ok=True, muted=state.muted)


@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    with state.lock:
        state.reconnect_requested = True
    return jsonify(ok=True)


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    with state.jpeg_lock:
        data = state.jpeg
    if not data:
        return jsonify(ok=False, error="no frame"), 503
    name = f"deer_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = SNAPSHOT_DIR / name
    path.write_bytes(data)
    return jsonify(ok=True, filename=name, url=f"/snapshots/{name}")


@app.route("/snapshots/<path:filename>")
def snapshots(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/api/scope/zoom", methods=["POST"])
def api_scope_zoom():
    data = request.get_json(force=True, silent=True) or {}
    if scope and scope.enabled:
        scope.set_zoom(int(data.get("index", 0)))
    return jsonify(ok=True)


@app.route("/api/scope/palette", methods=["POST"])
def api_scope_palette():
    data = request.get_json(force=True, silent=True) or {}
    if scope and scope.enabled:
        scope.set_palette(int(data.get("index", 0)))
    return jsonify(ok=True)


@app.route("/api/scope/image", methods=["POST"])
def api_scope_image():
    data = request.get_json(force=True, silent=True) or {}
    if scope and scope.enabled:
        scope.set_brightness_contrast(
            int(data.get("brightness", 50)),
            int(data.get("contrast", 50)),
        )
    return jsonify(ok=True)


def init_scope(host: str, demo: bool):
    global scope
    if demo:
        scope = ScopeControl(enabled=False)
        return
    scope = ScopeControl(host=host)

    def _connect():
        if scope.ping():
            print("Scope ISAPI connected — web sliders control device.")
            scope.sync_from_device()
        else:
            print("Scope not reachable — video only (connect to Taipan hotspot for ISAPI).")

    threading.Thread(target=_connect, daemon=True).start()


def main():
    global detector, alerts
    ap = argparse.ArgumentParser(description="AGM Taipan web deer scanner")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scope-host", default="10.15.12.1", help="Taipan ISAPI IP")
    ap.add_argument("--demo", action="store_true", help="Synthetic demo feed (no camera)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--model", help="YOLO .pt weights path")
    args = ap.parse_args()

    detector = make_detector(args.model)
    state.use_yolo = isinstance(detector, YoloDeerDetector)
    if state.use_yolo:
        state.yolo_conf = getattr(detector, "conf", 0.35)
    else:
        state.sensitivity = detector.state.sensitivity
    state.url = args.url
    state.demo = args.demo
    alerts = AlertManager(enabled=not args.no_audio)
    init_scope(args.scope_host, args.demo)

    t = threading.Thread(target=capture_loop, name="capture", daemon=True)
    t.start()

    url = f"http://{args.host}:{args.port}/"
    print(f"AGM Web Scanner running at {url}")
    if args.demo:
        print("Demo mode — synthetic thermal feed.")
    else:
        print("Connect laptop to Taipan hotspot for video.")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
