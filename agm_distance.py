"""Monocular distance estimation from deer bounding boxes (pinhole model)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEER_HEIGHT_M = 0.95
DEFAULT_FOCAL_PX = 520.0

TIER_BOUNDS = (("immediate", 25), ("near", 50), ("medium", 100))


@dataclass
class DistanceEstimate:
    meters: float
    label: str
    tier: str
    confidence: str


@dataclass
class DistanceEstimator:
    focal_px: float = DEFAULT_FOCAL_PX
    deer_height_m: float = DEFAULT_DEER_HEIGHT_M

    @classmethod
    def load(cls, path: Path | None = None) -> "DistanceEstimator":
        if path is None:
            path = Path(__file__).resolve().parent / "pi" / "distance_calib.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                focal_px=float(data.get("focal_px", DEFAULT_FOCAL_PX)),
                deer_height_m=float(data.get("deer_height_m", DEFAULT_DEER_HEIGHT_M)),
            )
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"focal_px": self.focal_px, "deer_height_m": self.deer_height_m}, indent=2),
            encoding="utf-8",
        )

    def from_bbox(self, bbox: tuple[int, int, int, int], zoom: float = 1.0) -> DistanceEstimate:
        _x, _y, _w, h = bbox
        h = max(h, 8)
        effective_focal = self.focal_px * max(zoom, 0.5)
        meters = (self.deer_height_m * effective_focal) / h
        meters = max(3.0, min(meters, 250.0))
        tier = "far"
        for name, bound in TIER_BOUNDS:
            if meters <= bound:
                tier = name
                break
        label = f"{int(round(meters))} m"
        conf = "high" if h >= 20 else "medium" if h >= 12 else "low"
        return DistanceEstimate(meters=meters, label=label, tier=tier, confidence=conf)

    def calibrate(self, bbox_height_px: int, distance_m: float) -> None:
        h = max(bbox_height_px, 1)
        self.focal_px = (distance_m * h) / self.deer_height_m


def main() -> None:
    ap = argparse.ArgumentParser(description="Distance calibrator for Pi deer scanner")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--bbox-height", type=int, help="bbox height in pixels at known distance")
    ap.add_argument("--distance-m", type=float, help="known distance in meters")
    ap.add_argument("--calib", type=Path, help="calibration JSON path")
    args = ap.parse_args()
    calib_path = args.calib or Path(__file__).resolve().parent / "pi" / "distance_calib.json"
    est = DistanceEstimator.load(calib_path)
    if args.calibrate:
        if not args.bbox_height or not args.distance_m:
            ap.error("--calibrate requires --bbox-height and --distance-m")
        est.calibrate(args.bbox_height, args.distance_m)
        est.save(calib_path)
        print(f"Saved focal_px={est.focal_px:.1f} to {calib_path}")
    else:
        demo = est.from_bbox((0, 0, 40, 80), zoom=1.0)
        print(f"Demo @ 80px bbox: {demo.label} ({demo.tier})")


if __name__ == "__main__":
    main()