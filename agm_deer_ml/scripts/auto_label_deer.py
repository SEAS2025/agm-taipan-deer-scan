"""
Auto-label deer in downloaded images using YOLO-World (open-vocabulary detection).

Writes YOLO .txt labels next to each image in dataset_internet/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
INTERNET = ROOT / "dataset_internet"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEER_PROMPTS = ["deer", "white-tailed deer", "whitetail deer"]


def label_folder(model, folder: Path, conf: float = 0.25) -> tuple[int, int]:
    if not folder.exists():
        return 0, 0
    labeled = skipped = 0
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.parent.name != "labels"]
    for img_path in imgs:
        lbl_path = img_path.with_suffix(".txt")
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            labeled += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue
        h, w = frame.shape[:2]

        results = model.predict(frame, conf=conf, verbose=False)
        lines: list[str] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if bw < 0.02 or bh < 0.02:
                    continue
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        if lines:
            labeled += 1
            print(f"  labeled {img_path.name} ({len(lines)} boxes)")
        else:
            skipped += 1
    return labeled, skipped


def copy_prelabeled_thermal(folder: Path) -> int:
    """Copy labels from DAID-T import subfolder if present."""
    lbl_dir = folder / "labels"
    if not lbl_dir.exists():
        return 0
    n = 0
    for lbl in lbl_dir.glob("*.txt"):
        for img in folder.glob(lbl.stem.replace("daid_", "") + ".*"):
            if img.suffix.lower() in IMG_EXTS:
                break
        else:
            img = folder / lbl.name.replace(".txt", ".jpg")
        target = folder / (Path(img.name).stem + ".txt") if img.exists() else None
        if target and not target.exists():
            target.write_text(lbl.read_text(encoding="utf-8"), encoding="utf-8")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO

    print("Loading YOLO-World for deer auto-labeling…")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(DEER_PROMPTS)

    for sub in ("visual", "thermal"):
        folder = INTERNET / sub
        print(f"\nLabeling {sub}/ …")
        labeled, skipped = label_folder(model, folder, args.conf)
        print(f"  {sub}: {labeled} labeled, {skipped} empty/skipped")


if __name__ == "__main__":
    main()
