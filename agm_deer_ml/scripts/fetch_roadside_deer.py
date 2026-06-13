"""
Download white-tailed deer images near roads — visual (RGB) and thermal (IR).

Sources (public APIs, no API keys):
  - iNaturalist (Odocoileus virginianus observations with photos)
  - Wikimedia Commons (roadside deer, thermal/infrared deer)

Output:
  agm_deer_ml/dataset_internet/visual/
  agm_deer_ml/dataset_internet/thermal/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_VISUAL = ROOT / "dataset_internet" / "visual"
OUT_THERMAL = ROOT / "dataset_internet" / "thermal"
USER_DAID = Path(r"C:\Users\User\agm_deer_ml")

HEADERS = {"User-Agent": "AGM-Deer-Scan/1.0 (research; contact: local)"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
        if len(data) < 8000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  skip {dest.name}: {e}")
        return False


def _safe_name(prefix: str, url: str, ext: str = ".jpg") -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return f"{prefix}_{h}{ext}"


def fetch_inaturalist(out: Path, taxon: str, term: str, max_n: int, prefix: str) -> int:
    saved = 0
    page = 1
    while saved < max_n:
        q = urllib.parse.urlencode(
            {
                "taxon_name": taxon,
                "term": term,
                "photos": "true",
                "quality_grade": "research,inat",
                "per_page": min(30, max_n - saved),
                "page": page,
            }
        )
        url = f"https://api.inaturalist.org/v1/observations?{q}"
        try:
            data = _get_json(url)
        except Exception as e:
            print(f"iNaturalist error: {e}")
            break
        results = data.get("results") or []
        if not results:
            break
        for obs in results:
            if saved >= max_n:
                break
            for photo in obs.get("photos") or []:
                if saved >= max_n:
                    break
                photo_url = photo.get("url") or ""
                if not photo_url:
                    continue
                photo_url = re.sub(r"/square\.", "/medium.", photo_url)
                photo_url = re.sub(r"/small\.", "/medium.", photo_url)
                fname = _safe_name(prefix, photo_url)
                dest = out / fname
                if dest.exists():
                    continue
                if _download(photo_url, dest):
                    saved += 1
                    print(f"  [{saved}/{max_n}] iNat {fname}")
                time.sleep(0.15)
        page += 1
        if page > 10:
            break
    return saved


def fetch_wikimedia(out: Path, search: str, max_n: int, prefix: str) -> int:
    saved = 0
    gsroffset = 0
    while saved < max_n:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": search,
                "gsrnamespace": "6",
                "gsrlimit": "20",
                "gsroffset": str(gsroffset),
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": "1280",
                "format": "json",
            }
        )
        url = f"https://commons.wikimedia.org/w/api.php?{params}"
        try:
            data = _get_json(url)
        except Exception as e:
            print(f"Wikimedia error: {e}")
            break
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            break
        for page in pages.values():
            if saved >= max_n:
                break
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue
            img_url = info.get("thumburl") or info.get("url")
            if not img_url:
                continue
            ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower() or ".jpg"
            if ext not in IMG_EXTS:
                ext = ".jpg"
            fname = _safe_name(prefix, img_url, ext)
            dest = out / fname
            if dest.exists():
                continue
            if _download(img_url, dest):
                saved += 1
                print(f"  [{saved}/{max_n}] Wiki {fname}")
            time.sleep(0.2)
        cont = (data.get("continue") or {}).get("gsroffset")
        if cont is None:
            break
        gsroffset = cont
    return saved


def import_daid_thermal(out: Path, max_n: int) -> int:
    """Copy thermal frames from existing local DAID-T dataset if present."""
    src = USER_DAID / "dataset" / "images" / "train"
    if not src.exists():
        src = ROOT / "dataset" / "images" / "train"
    if not src.exists():
        return 0
    saved = 0
    for img in sorted(src.glob("*")):
        if saved >= max_n:
            break
        if img.suffix.lower() not in IMG_EXTS:
            continue
        dest = out / f"daid_{img.name}"
        if dest.exists():
            saved += 1
            continue
        dest.write_bytes(img.read_bytes())
        lbl_src = USER_DAID / "dataset" / "labels" / "train" / (img.stem + ".txt")
        if not lbl_src.exists():
            lbl_src = ROOT / "dataset" / "labels" / "train" / (img.stem + ".txt")
        if lbl_src.exists():
            dest.with_suffix(".txt").write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")
        saved += 1
    if saved:
        print(f"  Imported {saved} local DAID-T thermal frames")
    return saved


def main():
    ap = argparse.ArgumentParser(description="Fetch roadside deer images from the internet")
    ap.add_argument("--max-visual", type=int, default=80)
    ap.add_argument("--max-thermal", type=int, default=40)
    args = ap.parse_args()

    OUT_VISUAL.mkdir(parents=True, exist_ok=True)
    OUT_THERMAL.mkdir(parents=True, exist_ok=True)

    print("=== Visual: white-tailed deer near roads ===")
    v = 0
    v += fetch_inaturalist(OUT_VISUAL, "Odocoileus virginianus", "road", args.max_visual // 2, "inat_road")
    v += fetch_inaturalist(OUT_VISUAL, "Odocoileus virginianus", "highway", args.max_visual // 4, "inat_hwy")
    v += fetch_wikimedia(OUT_VISUAL, "white-tailed deer road", args.max_visual - v, "wiki_road")
    v += fetch_wikimedia(OUT_VISUAL, "deer crossing road wildlife", max(0, args.max_visual - v), "wiki_xing")

    print("\n=== Thermal / infrared deer ===")
    t = 0
    t += fetch_wikimedia(OUT_THERMAL, "thermal imaging deer", args.max_thermal // 3, "wiki_thermal")
    t += fetch_wikimedia(OUT_THERMAL, "infrared deer wildlife", args.max_thermal // 3, "wiki_ir")
    t += fetch_wikimedia(OUT_THERMAL, "FLIR deer", max(0, args.max_thermal // 3 - t), "wiki_flir")
    t += import_daid_thermal(OUT_THERMAL, max(0, args.max_thermal - t))

    print(f"\nDone: {v} visual, {t} thermal -> {ROOT / 'dataset_internet'}")


if __name__ == "__main__":
    main()
