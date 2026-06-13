"""
Train a small YOLOv8n model on thermal wildlife (DAID-T) for deer/animal detection.

CPU-friendly defaults: nano model, 640px, 30 epochs.
Run from agm_deer_ml/:
  python scripts/prepare_dataset.py --clean
  python scripts/train_deer_yolo.py
  python scripts/train_deer_yolo.py --epochs 50 --imgsz 416
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"
DEFAULT_MODEL = "yolov8n.pt"


def write_data_yaml():
    content = f"""# Thermal wildlife — DAID-T (drone thermal animals) + custom frames
path: {ROOT.as_posix()}
train: dataset/images/train
val: dataset/images/val

nc: 1
names:
  0: deer
"""
    DATA_YAML.write_text(content, encoding="utf-8")
    print(f"Wrote {DATA_YAML}")


def main():
    ap = argparse.ArgumentParser(description="Train small YOLO deer detector (local CPU/GPU)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8, help="reduce to 4 if OOM on CPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="yolov8n.pt (smallest) recommended")
    ap.add_argument("--device", default="", help="cuda:0 or cpu (auto if blank)")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="deer_thermal")
    ap.add_argument("--data", default="", help="explicit data.yaml (overrides default DAID-T set)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--no-aug", action="store_true", help="disable extra vehicle-motion augmentation")
    args = ap.parse_args()

    if args.data:
        data_yaml = Path(args.data)
        if not data_yaml.exists():
            raise SystemExit(f"--data not found: {data_yaml}")
    else:
        train_img = ROOT / "dataset" / "images" / "train"
        if not train_img.exists() or not any(train_img.iterdir()):
            raise SystemExit("No training images. Run: python scripts/prepare_dataset.py --clean")
        write_data_yaml()
        data_yaml = DATA_YAML

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Training {args.model} on {data_yaml} ({args.epochs} epochs, imgsz={args.imgsz})")

    # Augmentation tuned to simulate a moving-vehicle deployment: mosaic mixes
    # backgrounds, blur/motion approximate vehicle vibration + speed, and the
    # dataset already includes polarity-inverted copies for white/black-hot.
    aug = {} if args.no_aug else dict(
        mosaic=1.0, mixup=0.15, close_mosaic=10,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,   # grayscale: only brightness jitter
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, erasing=0.3,
    )

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
        **aug,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    deploy = ROOT / "models" / "deer_thermal_best.pt"
    deploy.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil
        shutil.copy2(best, deploy)
        print(f"\nBest weights copied to: {deploy}")
        print(f"Use in scanner: python agm_deer_scanner.py --model {deploy}")
    return results


if __name__ == "__main__":
    main()
