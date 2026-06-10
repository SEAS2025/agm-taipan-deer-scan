@echo off
cd /d "%~dp0"
echo Preparing dataset...
python scripts\prepare_dataset.py --clean
echo Training YOLOv8n (CPU)...
python scripts\train_deer_yolo.py --epochs 30 --batch 4 --imgsz 640
echo Done. Weights: models\deer_thermal_best.pt
pause
