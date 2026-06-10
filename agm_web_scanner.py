"""
Lightweight local web scanner for AGM Taipan thermal deer detection.

  python agm_web_scanner.py
  python agm_web_scanner.py --port 8080

Open http://127.0.0.1:8080 in a browser (auto-opens on start).
Connect laptop to Taipan hotspot before starting.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from agm_deer_scanner import (
    DEER_PROFILE,
    AlertManager,
    Detection,
    YoloDeerDetector,
    make_detector,
)

DEFAULT_URL = "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101"

app = Flask(__name__)


@dataclass
class ScanState:
    url: str = DEFAULT_URL
    sensitivity: float = 1.0
    yolo_conf: float = 0.35
    use_yolo: bool = False
    muted: bool = False
    status: str = "STARTING"
    fps: float = 0.0
    armed: bool = False
    detections: int = 0
    connected: bool = False
    reconnect_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    jpeg: Optional[bytes] = None
    jpeg_lock: threading.Lock = field(default_factory=threading.Lock)


state = ScanState()
detector = None  # set in main
alerts: Optional[AlertManager] = None


def draw_hud(frame: np.ndarray, dets: list[Detection], status: str, fps: float, armed: bool, sens: float):
    out = frame.copy()
    h, w = out.shape[:2]
    for d in dets:
        x, y, bw, bh = d.bbox
        is_deer = d.score >= DEER_PROFILE["score_threshold"]
        color = (0, 0, 255) if is_deer else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3 if is_deer else 1)
        cv2.putText(out, f"{d.label} {d.score:.0%}", (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)
    bar_color = (0, 0, 255) if armed else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(out, f"DEER SCAN | {status} | {fps:.1f} FPS | sens {sens:.1f}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
    if armed:
        cv2.putText(out, "!!! DEER !!!", (w // 2 - 120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


def open_capture(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_loop():
    global detector, alerts
    cap = None
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
            if state.reconnect_requested:
                state.reconnect_requested = False
                if cap:
                    cap.release()
                    cap = None

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
            if time.time() - last_ok > 3:
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

        with state.lock:
            sens = state.yolo_conf if state.use_yolo else state.sensitivity

        if frame_i % 2 == 0:
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
                    alerts.muted = state.muted
                    alerts.trigger(len(deer_hits))
                confirm_streak = 0
            elif deer_hits:
                status = f"TRACKING ({len(deer_hits)})"
            else:
                status = "SCANNING"
            display = draw_hud(frame, dets, status, fps, armed, sens)
            with state.lock:
                state.status = status
                state.fps = fps
                state.armed = armed
                state.detections = len(dets)
        else:
            with state.lock:
                status = state.status
                armed = state.armed
            display = draw_hud(frame, [], status, fps, armed, sens)

        ok_enc, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok_enc:
            with state.jpeg_lock:
                state.jpeg = buf.tobytes()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AGM Deer Scan</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Segoe UI, sans-serif; background: #1a1a1a; color: #eee; }
  header { padding: 12px 16px; background: #0d3d0d; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  header h1 { margin: 0; font-size: 1.1rem; flex: 1; }
  .badge { padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
  .badge.scan { background: #2d5a2d; }
  .badge.alert { background: #8b0000; animation: pulse 0.8s infinite; }
  @keyframes pulse { 50% { opacity: 0.7; } }
  main { display: grid; grid-template-columns: 1fr 280px; gap: 0; min-height: calc(100vh - 52px); }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .video-wrap { background: #000; display: flex; align-items: center; justify-content: center; min-height: 400px; }
  .video-wrap img { max-width: 100%; max-height: calc(100vh - 120px); object-fit: contain; }
  aside { padding: 16px; background: #252525; border-left: 1px solid #333; }
  label { display: block; margin: 12px 0 6px; font-size: 0.9rem; color: #aaa; }
  input[type=range] { width: 100%; }
  .val { font-size: 1.4rem; font-weight: 600; color: #6f6; }
  button { width: 100%; margin-top: 8px; padding: 12px; font-size: 1rem; border: none; border-radius: 6px; cursor: pointer; }
  .btn-mute { background: #444; color: #fff; }
  .btn-mute.on { background: #633; }
  .btn-reconnect { background: #335; color: #fff; }
  .stats { margin-top: 20px; font-size: 0.85rem; color: #999; line-height: 1.8; }
  .hint { margin-top: 16px; font-size: 0.75rem; color: #666; }
</style>
</head>
<body>
<header>
  <h1>AGM Taipan — Deer Scan</h1>
  <span id="statusBadge" class="badge scan">SCANNING</span>
  <span id="fps">0 FPS</span>
</header>
<main>
  <div class="video-wrap">
    <img id="feed" src="/video.mjpg" alt="Live thermal feed">
  </div>
  <aside>
    <label>Sensitivity <span class="val" id="sensVal">1.0</span></label>
    <input type="range" id="sens" min="0" max="100" value="33">
    <p style="font-size:0.8rem;color:#888;margin:4px 0 0">Lower = fewer alerts</p>
    <button class="btn-mute" id="muteBtn">Mute audio</button>
    <button class="btn-reconnect" id="reconnectBtn">Reconnect stream</button>
    <div class="stats">
      <div>Detections: <span id="detCount">0</span></div>
      <div>Stream: <span id="conn">…</span></div>
    </div>
    <p class="hint">Connect to Taipan hotspot first. Press F11 for fullscreen.</p>
  </aside>
</main>
<script>
const sens = document.getElementById('sens');
const sensVal = document.getElementById('sensVal');
const muteBtn = document.getElementById('muteBtn');
let muted = false;

function sensFromSlider(v) { return 0.5 + (v / 100) * 1.5; }
function sliderFromSens(s) { return Math.round(((s - 0.5) / 1.5) * 100); }

sens.addEventListener('input', () => {
  const v = sensFromSlider(+sens.value);
  sensVal.textContent = v.toFixed(1);
  fetch('/api/sensitivity', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({value: v}) });
});

muteBtn.onclick = () => {
  muted = !muted;
  muteBtn.textContent = muted ? 'Unmute audio' : 'Mute audio';
  muteBtn.classList.toggle('on', muted);
  fetch('/api/mute', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({muted}) });
};

document.getElementById('reconnectBtn').onclick = () => fetch('/api/reconnect', {method: 'POST'});

async function poll() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    document.getElementById('fps').textContent = j.fps.toFixed(1) + ' FPS';
    document.getElementById('detCount').textContent = j.detections;
    document.getElementById('conn').textContent = j.connected ? 'Connected' : 'Reconnecting…';
    const b = document.getElementById('statusBadge');
    b.textContent = j.status;
    b.className = 'badge ' + (j.armed ? 'alert' : 'scan');
    if (j.use_yolo) sensVal.textContent = j.sensitivity.toFixed(2) + ' conf';
    else sensVal.textContent = j.sensitivity.toFixed(1);
  } catch (e) {}
}
setInterval(poll, 1000);
poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/video.mjpg")
def video():
    def gen():
        while True:
            with state.jpeg_lock:
                frame = state.jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state.lock:
        sens = state.yolo_conf if state.use_yolo else state.sensitivity
        return jsonify(
            status=state.status,
            fps=state.fps,
            armed=state.armed,
            detections=state.detections,
            connected=state.connected,
            sensitivity=sens,
            muted=state.muted,
            use_yolo=state.use_yolo,
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


def main():
    global detector, alerts
    ap = argparse.ArgumentParser(description="AGM Taipan web deer scanner")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
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
    alerts = AlertManager(enabled=not args.no_audio)

    t = threading.Thread(target=capture_loop, name="capture", daemon=True)
    t.start()

    url = f"http://{args.host}:{args.port}/"
    print(f"AGM Web Scanner running at {url}")
    print("Connect laptop to Taipan hotspot for video.")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
