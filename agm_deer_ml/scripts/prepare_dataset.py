"""
Prepare DAID-T thermal wildlife dataset (YOLO format) for local training.

Consolidates extracted zip contents into:
  agm_deer_ml/dataset/images/{train,val}
  agm_deer_ml/dataset/labels/{train,val}

Also supports adding custom frames from a folder or RTSP capture.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset_raw"
INTERNET = ROOT / "dataset_internet"
USER_DAID = Path(r"C:\Users\User\agm_deer_ml")
OUT = ROOT / "dataset"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS or "__MACOSX" in img.parts:
            continue
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def copy_pairs(pairs: list[tuple[Path, Path]], split: str):
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        dst_img = img_dir / img.name
        dst_lbl = lbl_dir / lbl.name
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, dst_lbl)


def add_custom_folder(custom: Path, split: str = "train", val_ratio: float = 0.15):
    """Copy custom images; create empty label if none (background frames)."""
    imgs = [p for p in custom.rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) > 5 else 0
    for i, img in enumerate(imgs):
        sp = "val" if i < n_val else split
        img_dir = OUT / "images" / sp
        lbl_dir = OUT / "labels" / sp
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, img_dir / img.name)
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            shutil.copy2(lbl, lbl_dir / lbl.name)
        else:
            (lbl_dir / (img.stem + ".txt")).write_text("", encoding="utf-8")


def import_internet_labeled():
    """Merge auto-labeled internet visual + thermal images."""
    n = 0
    for sub in ("visual", "thermal"):
        folder = INTERNET / sub
        if not folder.exists():
            continue
        for img in folder.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS or img.parent.name == "labels":
                continue
            lbl = img.with_suffix(".txt")
            if not lbl.exists():
                continue
            sp = "val" if n % 5 == 0 else "train"
            img_dir = OUT / "images" / sp
            lbl_dir = OUT / "labels" / sp
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            stem = f"inet_{sub}_{img.stem}"
            shutil.copy2(img, img_dir / f"{stem}{img.suffix.lower()}")
            shutil.copy2(lbl, lbl_dir / f"{stem}.txt")
            n += 1
    return n


def import_user_daid():
    """Import existing local DAID-T dataset if repo dataset_raw is empty."""
    if any((RAW / p).exists() for p in ("train_part_1", "train_part_2", "val")):
        return 0
    src_train = USER_DAID / "dataset" / "images" / "train"
    src_val = USER_DAID / "dataset" / "images" / "val"
    if not src_train.exists():
        return 0
    n = 0
    for split, src in (("train", src_train), ("val", src_val)):
        lbl_src = USER_DAID / "dataset" / "labels" / split
        if not src.exists():
            continue
        for img in src.glob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_src / (img.stem + ".txt")
            if not lbl.exists():
                continue
            img_dir = OUT / "images" / split
            lbl_dir = OUT / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)
            n += 1
    if n:
        print(f"Imported {n} frames from local DAID-T ({USER_DAID})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", help="folder of extra thermal frames (optional labels)")
    ap.add_argument("--clean", action="store_true", help="wipe dataset/ before prepare")
    ap.add_argument("--merge-internet", action="store_true", help="include dataset_internet/")
    args = ap.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)

    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []

    for part in ["train_part_1", "train_part_2"]:
        p = RAW / part
        if p.exists():
            train_pairs.extend(find_pairs(p))

    val_root = RAW / "val"
    if val_root.exists():
        val_pairs.extend(find_pairs(val_root))

    # De-dup by filename
    def dedup(pairs):
        seen = set()
        out = []
        for img, lbl in pairs:
            if img.name in seen:
                continue
            seen.add(img.name)
            out.append((img, lbl))
        return out

    train_pairs = dedup(train_pairs)
    val_pairs = dedup(val_pairs)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    if args.custom:
        add_custom_folder(Path(args.custom))

    import_user_daid()
    if args.merge_internet:
        inet_n = import_internet_labeled()
        print(f"Merged {inet_n} internet images (visual + thermal)")

    n_train = len(list((OUT / "images" / "train").glob("*")))
    n_val = len(list((OUT / "images" / "val").glob("*")))
    print(f"Dataset ready: {n_train} train, {n_val} val")
    print(f"  images: {OUT / 'images'}")
    print(f"  labels: {OUT / 'labels'}")


if __name__ == "__main__":
    main()
