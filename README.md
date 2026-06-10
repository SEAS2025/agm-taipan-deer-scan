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
python agm_web_scanner.py --model agm_deer_ml/models/deer_thermal_best.pt
python agm_web_scanner.py --no-audio --no-browser
```

## Train a YOLO deer model

See [agm_deer_ml/README.md](agm_deer_ml/README.md) for dataset prep and training on CPU/GPU.

## Project layout

```
agm_web_scanner.py          # Browser-based scanner (Flask)
agm_deer_scanner.py         # Desktop OpenCV scanner + shared detection logic
agm_taipan_stream_explorer.py
agm_deer_ml/                # YOLO training scripts and docs
requirements.txt
```

## Requirements

- Python 3.10+
- OpenCV with FFmpeg (RTSP)
- Windows recommended for TTS/beep alerts (web scanner works cross-platform)
