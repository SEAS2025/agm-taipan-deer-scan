# Raspberry Pi Edge Deployment

Low-latency deer detection + distance audio for ambulance-mounted AGM Taipan.

## Architecture

Taipan RTSP -> Pi grab thread (drop-old frames) -> YOLO @ 416px -> distance -> tiered audio

No Flask — target end-to-end latency under 150 ms on Pi 5.

## Quick start

1. Train on laptop: `python agm_deer_ml/scripts/train_deer_yolo.py`
2. Export: `python agm_deer_ml/scripts/export_pi_model.py`
3. Copy repo to Pi, `pip3 install -r pi/requirements-pi.txt`
4. Calibrate: `python3 agm_distance.py --calibrate --bbox-height 80 --distance-m 50`
5. Run: `python3 agm_pi_scanner.py --model agm_deer_ml/models/deer_thermal_best.pt --headless`

See pi/deer-scan.service for systemd auto-start.
