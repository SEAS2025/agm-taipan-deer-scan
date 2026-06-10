# AGM Deer ML — local thermal deer detector training

Small **YOLOv8n** model trained on drone thermal wildlife footage (DAID-T) and your own clips.

## Quick start

```powershell
cd C:\Users\User\agm_deer_ml

# 1. Dataset already downloaded (DAID-T). Re-prepare if needed:
python scripts\prepare_dataset.py --clean

# 2. Train (CPU ~30–90 min for 30 epochs; GPU much faster)
python scripts\train_deer_yolo.py --epochs 30 --batch 4

# 3. Run live scanner with trained weights
python C:\Users\User\agm_deer_scanner.py --model models\deer_thermal_best.pt
```

## Add your own thermal deer video

```powershell
# From local MP4
python scripts\extract_frames.py --video C:\path\to\thermal_deer.mp4 --out custom_frames --every 15

# From Taipan RTSP (scope on hotspot)
python scripts\extract_frames.py --rtsp "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101" --out custom_frames --seconds 120

# Label boxes (class 0) in Roboflow or Label Studio, save as YOLO .txt next to each .jpg
# Then merge into training set:
python scripts\prepare_dataset.py --custom custom_frames
python scripts\train_deer_yolo.py --epochs 40
```

## Layout

```
agm_deer_ml/
  dataset/images/{train,val}   # 853 + 367 thermal frames (DAID-T)
  dataset/labels/{train,val}   # YOLO format, class 0 = deer/animal
  models/deer_thermal_best.pt    # copied after training
  runs/deer_thermal/             # ultralytics outputs
  scripts/
    prepare_dataset.py
    extract_frames.py
    train_deer_yolo.py
  data.yaml                      # auto-generated
```

## Dataset source

**DAID-T** — Drone Animal Image Dataset (thermal), Nguyen et al.  
GitHub: `dtnguyen0304/Drone-based-wildlife-monitoring`  
Single class `animal` relabeled as `deer` for roadside thermal use.

## Training tips

| Setting | CPU (no GPU) | With NVIDIA GPU |
|---------|--------------|-----------------|
| Model | `yolov8n.pt` | `yolov8n.pt` or `yolov8s.pt` |
| Batch | 4 | 16 |
| Epochs | 30 | 50+ |
| imgsz | 640 | 640 |

Fine-tune on your AGM Taipan frames after base training for best roadside results.
