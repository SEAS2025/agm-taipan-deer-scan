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
    Detection,
    YoloDeerDetector,
    make_detector,
)
from agm_deer_alert import assess_nearest_threat
from agm_distance import DistanceEstimator
from agm_pi_audio import PiAudioAlerts
from agm_scope_control import ScopeControl
from agm_llm import LLMAssistant
from agm_training_manager import get_status, is_running, start_pipeline

DEFAULT_URL = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"
WEB_DIR = Path(__file__).resolve().parent / "web"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
FEEDER_DIR = SNAPSHOT_DIR / "feeder"
FEEDER_LOG = Path(__file__).resolve().parent / "agm_deer_ml" / "runs" / "feeder_session.json"

app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")

llm = LLMAssistant()


@dataclass
class ScanState:
    url: str = DEFAULT_URL
    demo: bool = False
    sensitivity: float = 1.0
    yolo_conf: float = 0.35
    use_yolo: bool = False
    muted: bool = False
    detection_enabled: bool = True
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
alerts: Optional[PiAudioAlerts] = None
dist_estimator = DistanceEstimator.load()
scope: Optional[ScopeControl] = None
snapshot_every_s: float = 0.0
feeder_mode: bool = False
_last_snapshot_t = 0.0
_last_armed = False


def draw_hud(
    frame: np.ndarray,
    dets: list[Detection],
    status: str,
    fps: float,
    armed: bool,
    sens: float,
    eff_threshold: float | None = None,
    detection_on: bool = True,
):
    out = frame.copy()
    h, w = out.shape[:2]
    thresh = eff_threshold if eff_threshold is not None else DEER_PROFILE["score_threshold"]
    for d in dets:
        x, y, bw, bh = d.bbox
        is_deer = d.score >= thresh
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
    bar_color = (0, 0, 255) if armed else ((100, 100, 100) if not detection_on else (0, 200, 0))
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    sens_label = f"sens {sens:.1f}" if detection_on else "DETECTION OFF"
    cv2.putText(
        out,
        f"DEER SCAN | {status} | {fps:.1f} FPS | {sens_label}",
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
        use_yolo = state.use_yolo
        detection_on = state.detection_enabled

    if not detection_on:
        display = draw_hud(frame, [], "DETECTION OFF", 0.0, False, sens, detection_on=False)
        return display, "DETECTION OFF", False, 0, 0, 0

    dets = detector.detect(frame)
    if use_yolo:
        eff = sens
        deer_hits = [d for d in dets if d.score >= eff]
    else:
        eff = score_threshold / max(sens, 0.35)
        deer_hits = [d for d in dets if d.score >= eff]

    if deer_hits:
        confirm_streak += 1
    else:
        confirm_streak = max(0, confirm_streak - 1)

    armed = confirm_streak >= DEER_PROFILE["confirm_frames"]
    if armed:
        status = f"DEER ALERT ({len(deer_hits)})"
        if alerts and deer_hits:
            fh, fw = frame.shape[:2]
            threat = assess_nearest_threat(deer_hits, fw, dist_estimator)
            if threat:
                with state.lock:
                    alerts.muted = state.muted
                alerts.trigger_threat(threat)
        confirm_streak = 0
    elif deer_hits:
        status = f"TRACKING ({len(deer_hits)})"
    else:
        status = "SCANNING"

    display = draw_hud(frame, dets, status, 0.0, armed, sens, eff_threshold=eff)
    return display, status, armed, len(dets), len(deer_hits), confirm_streak


def _write_feeder_log(name: str, reason: str, armed: bool, det_n: int, deer_n: int) -> None:
    import json

    FEEDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "filename": name,
        "reason": reason,
        "armed": armed,
        "detections": det_n,
        "deer_hits": deer_n,
        "fps": state.fps,
        "connected": state.connected,
        "status": state.status,
    }
    FEEDER_LOG.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def save_snapshot(reason: str = "manual", *, armed: bool = False, det_n: int = 0, deer_n: int = 0) -> Optional[str]:
    """Save current HUD frame; returns filename or None."""
    with state.jpeg_lock:
        data = state.jpeg
    if not data:
        return None
    dest = FEEDER_DIR if feeder_mode else SNAPSHOT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    name = f"feeder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg" if feeder_mode else f"deer_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    (dest / name).write_bytes(data)
    if feeder_mode:
        _write_feeder_log(name, reason, armed, det_n, deer_n)
    return name


def capture_loop():
    global detector, alerts, _last_snapshot_t, _last_armed
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
                det_on = state.detection_enabled
            if det_on:
                display = draw_hud(frame, [], status, fps, armed, sens)
            else:
                display = draw_hud(frame, [], "DETECTION OFF", fps, False, sens, detection_on=False)

        ok_enc, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok_enc:
            with state.jpeg_lock:
                state.jpeg = buf.tobytes()

        if snapshot_every_s > 0 and time.time() - _last_snapshot_t >= snapshot_every_s:
            with state.lock:
                armed_now = state.armed
                det_n = state.detections
                deer_n = state.deer_hits
            save_snapshot("interval", armed=armed_now, det_n=det_n, deer_n=deer_n)
            _last_snapshot_t = time.time()

        with state.lock:
            armed_now = state.armed
            det_n = state.detections
            deer_n = state.deer_hits
        if armed_now and not _last_armed:
            save_snapshot("deer_alert", armed=True, det_n=det_n, deer_n=deer_n)
        _last_armed = armed_now


@app.route("/audio/deer_deer.wav")
def deer_audio():
    clip_dir = Path(__file__).resolve().parent / "audio" / "egpws"
    return send_from_directory(clip_dir, "deer_deer.wav")


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


def _scan_context() -> dict:
    with state.lock:
        ctx = {
            "status": state.status,
            "fps": state.fps,
            "armed": state.armed,
            "detections": state.detections,
            "deer_hits": state.deer_hits,
            "connected": state.connected,
            "use_yolo": state.use_yolo,
            "demo": state.demo,
        }
    ctx["training"] = get_status()
    ctx["llm"] = llm.status()
    return ctx


@app.route("/api/llm/status")
def api_llm_status():
    return jsonify(llm.status())


@app.route("/api/llm/chat", methods=["POST"])
def api_llm_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(ok=False, error="empty message"), 400
    result = llm.chat(message, _scan_context())
    return jsonify(result)


@app.route("/api/llm/analyze", methods=["POST"])
def api_llm_analyze():
    with state.jpeg_lock:
        frame = state.jpeg
    if not frame:
        return jsonify(ok=False, error="no frame"), 503
    result = llm.analyze_frame(frame, _scan_context())
    return jsonify(result)


@app.route("/api/training/status")
def api_training_status():
    return jsonify(get_status())


@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    data = request.get_json(force=True, silent=True) or {}
    result = start_pipeline(
        epochs=int(data.get("epochs", 30)),
        batch=int(data.get("batch", 4)),
        max_visual=int(data.get("max_visual", 80)),
        max_thermal=int(data.get("max_thermal", 40)),
    )
    return jsonify(result)


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
            detection_enabled=state.detection_enabled,
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
            state.sensitivity = max(0.35, min(3.0, val))
            detector.state.sensitivity = state.sensitivity
    return jsonify(ok=True, sensitivity=state.sensitivity if not state.use_yolo else state.yolo_conf)


@app.route("/api/detection", methods=["POST"])
def api_detection():
    data = request.get_json(force=True, silent=True) or {}
    with state.lock:
        state.detection_enabled = bool(data.get("enabled", True))
        if not state.detection_enabled:
            state.armed = False
            state.deer_hits = 0
            state.detections = 0
            state.status = "DETECTION OFF"
    return jsonify(ok=True, enabled=state.detection_enabled)


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
    with state.lock:
        armed = state.armed
        det_n = state.detections
        deer_n = state.deer_hits
    name = save_snapshot("manual", armed=armed, det_n=det_n, deer_n=deer_n)
    if not name:
        return jsonify(ok=False, error="no frame"), 503
    sub = "feeder" if feeder_mode else ""
    url = f"/snapshots/feeder/{name}" if sub else f"/snapshots/{name}"
    return jsonify(ok=True, filename=name, url=url)


@app.route("/api/frame.jpg")
def api_frame_jpg():
    with state.jpeg_lock:
        data = state.jpeg
    if not data:
        return jsonify(ok=False, error="no frame"), 503
    return Response(data, mimetype="image/jpeg")


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
        idx = int(data.get("index", 0))
        scope.set_palette(idx)
        return jsonify(ok=True, palette_index=idx, palette_name=scope.palette_name)
    return jsonify(ok=False, error="scope unavailable")


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
    ap.add_argument("--auto-train", action="store_true", help="Start dataset fetch + training on launch")
    ap.add_argument("--train-epochs", type=int, default=30)
    ap.add_argument("--feeder", action="store_true", help="Tripod feeder watch: auto-snapshots to snapshots/feeder/")
    ap.add_argument("--snapshot-every", type=float, default=0, metavar="SEC", help="Auto-save JPEG every N seconds (60 with --feeder)")
    args = ap.parse_args()

    global snapshot_every_s, feeder_mode
    feeder_mode = args.feeder
    snapshot_every_s = args.snapshot_every or (60.0 if args.feeder else 0.0)

    detector = make_detector(args.model)
    state.use_yolo = isinstance(detector, YoloDeerDetector)
    if state.use_yolo:
        state.yolo_conf = getattr(detector, "conf", 0.35)
    else:
        state.sensitivity = detector.state.sensitivity
    state.url = args.url
    state.demo = args.demo
    alerts = PiAudioAlerts(enabled=not args.no_audio, voice=True)
    init_scope(args.scope_host, args.demo)

    t = threading.Thread(target=capture_loop, name="capture", daemon=True)
    t.start()

    if args.auto_train and not is_running():
        print("Starting training pipeline (fetch internet deer images + YOLO train)…")
        start_pipeline(epochs=args.train_epochs, batch=4)

    url = f"http://{args.host}:{args.port}/"
    print(f"AGM Web Scanner running at {url}")
    print(f"Text assistant backend: {llm.backend}")
    if args.demo:
        print("Demo mode — synthetic thermal feed.")
    else:
        print("Connect laptop to Taipan hotspot for video.")
    if feeder_mode:
        print(f"Feeder mode — snapshots every {snapshot_every_s:.0f}s -> {FEEDER_DIR}")
        print(f"Session log: {FEEDER_LOG}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
