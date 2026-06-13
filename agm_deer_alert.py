"""Deer threat assessment (side/distance for display; audio is warning + Deer)."""
from __future__ import annotations

from dataclasses import dataclass

from agm_deer_scanner import Detection
from agm_distance import DistanceEstimate, DistanceEstimator


@dataclass
class DeerThreat:
    side: str
    distance: DistanceEstimate
    detection: Detection
    count: int

    @property
    def tier(self) -> str:
        return self.distance.tier

    @property
    def meters(self) -> float:
        return self.distance.meters


def side_of_road(centroid_x: int, frame_width: int, left_frac: float = 0.38, right_frac: float = 0.62) -> str:
    if frame_width <= 0:
        return "ahead"
    rel = centroid_x / frame_width
    if rel < left_frac:
        return "left"
    if rel > right_frac:
        return "right"
    return "ahead"


def format_terrain_callout(threat: DeerThreat) -> str:
    side = threat.side.upper()
    m = int(round(threat.meters))
    if threat.count > 1:
        return f"Deer — {threat.count} detected, nearest {side} {m}m"
    return f"Deer — {side} {m}m"


def assess_nearest_threat(detections, frame_width, estimator, zoom=1.0):
    if not detections:
        return None
    best = None
    for d in detections:
        dist = estimator.from_bbox(d.bbox, zoom)
        side = side_of_road(d.centroid[0], frame_width)
        threat = DeerThreat(side=side, distance=dist, detection=d, count=len(detections))
        if best is None or dist.meters < best.meters:
            best = threat
    return best