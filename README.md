# AGM Taipan Deer Scan

Live thermal deer detection for the **AGM Taipan** scope — desktop OpenCV viewer and browser-based web scanner for ambulance-mounted roadside use.

## Features

- **Web scanner** — Flask UI with MJPEG live feed, sensitivity slider, mute, and reconnect
- **Desktop scanner** — OpenCV window with keyboard controls and voice alerts
- **Thermal heuristic detector** — works out of the box (no model download)
- **Optional YOLO model** — train a small YOLOv8n on thermal wildlife data (see `agm_deer_ml/`)

## Quick start

1. Connect your laptop to the **Taipan WiFi hotspot**
2. Install dependencies:

```powershell
cd agm-taipan-deer-scan
pip install -r requirements.txt
```

3. Run the web scanner:

```powershell
python agm_web_scanner.py
```

Open http://127.0.0.1:8080 in your browser.

**Try without hardware** (demo mode with synthetic thermal feed):

```powershell
python agm_web_scanner.py --demo --no-browser
```

Or run the desktop scanner:

```powershell
python agm_deer_scanner.py
```

## RTSP stream

Default stream URL (edit in the scripts if your credentials differ):

```
rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101
```

Use `agm_taipan_stream_explorer.py --probe-rtsp` to discover alternate RTSP paths on your device.

## Web scanner options

```powershell
python agm_web_scanner.py --port 8080
python agm_web_scanner.py --demo                    # UI test without Taipan
python agm_web_scanner.py --model agm_deer_ml/models/deer_thermal_best.pt
python agm_web_scanner.py --no-audio --no-browser
```

The web UI includes live MJPEG video, sensitivity control, scope zoom/palette/brightness (via ISAPI), snapshots, and browser + server audio alerts.

## Train a YOLO deer model

See [agm_deer_ml/README.md](agm_deer_ml/README.md) for dataset prep and training on CPU/GPU.

## YEE TS3-19 scope (separate from Taipan)

White-label **TS Series** thermal scope (384×288, 19 mm). WiFi SSID `XfdAp…`, open network, **Cam802** app — not RTSP like the Taipan.

See **[yee/README.md](yee/README.md)** for manuals, pairing, USB notes, and PC streaming experiments.

```powershell
launch_yee_probe.bat          # probe scope WiFi (join XfdAp... first)
launch_yee_probe.bat listen 60  # capture UDP while Cam802 runs on phone
```

## Project layout

```
agm_pi_scanner.py           # Pi edge scanner (low latency + distance audio)
agm_distance.py             # Monocular distance estimation
agm_pi_audio.py             # Tiered Pi audio cues
agm_web_scanner.py          # Browser-based scanner (Flask)
agm_deer_scanner.py         # Desktop OpenCV scanner + shared detection logic
agm_scope_control.py        # ISAPI scope control (zoom, palette, image)
yee_stream_probe.py         # YEE scope WiFi probe / UDP listener
yee/                          # YEE TS3-19 docs + offline manuals (PDF)
pi/                         # Pi deployment docs + systemd service
agm_deer_ml/                # YOLO training scripts and docs
requirements.txt
```

## Requirements

- Python 3.10+
- OpenCV with FFmpeg (RTSP)
- Windows recommended for TTS/beep alerts (web scanner works cross-platform)
