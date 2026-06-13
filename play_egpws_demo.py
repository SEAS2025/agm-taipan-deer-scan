"""Play MK-VIII matched deer deer alert."""
from __future__ import annotations

import argparse
import time

from agm_deer_alert import DeerThreat, format_terrain_callout
from agm_deer_scanner import Detection
from agm_distance import DistanceEstimate
from agm_pi_audio import PiAudioAlerts

DET = Detection((10, 10, 30, 50), 0.92, "DEER", (25, 35))
THREAT = DeerThreat("left", DistanceEstimate(40, "40 m", "near", "high"), DET, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play deer deer EGPWS alert")
    ap.add_argument("--compare", action="store_true", help="Play terrain.wav then deer_deer.wav")
    args = ap.parse_args()
    audio = PiAudioAlerts(enabled=True, voice=True)
    if args.compare:
        from agm_egpws_voice import play_wav
        from pathlib import Path
        clips = Path(__file__).resolve().parent / "audio" / "egpws"
        print("Reference: terrain.wav")
        play_wav(clips / "terrain.wav")
        time.sleep(0.3)
        print("Matched: deer_deer.wav")
    print(format_terrain_callout(THREAT))
    audio.play_sync(THREAT)
    print("Done.")


if __name__ == "__main__":
    main()