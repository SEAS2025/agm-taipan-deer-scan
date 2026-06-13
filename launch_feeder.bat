@echo off
title AGM Feeder Watch
cd /d "%~dp0"
echo ============================================================
echo  Feeder watch — tripod thermal at deer feeder
echo  1. Mount scope on TRIPOD (must stay still for auto-label)
echo  2. Connect laptop to Taipan WiFi hotspot
echo  3. Open http://127.0.0.1:8080
echo  Snapshots: snapshots\feeder\  (every 60s + on deer alert)
echo ============================================================
python agm_web_scanner.py --feeder --model agm_deer_ml\models\deer_thermal_best.pt --no-browser
pause