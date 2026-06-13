"""Deer deer EGPWS callout playback."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from agm_deer_alert import DeerThreat
from agm_egpws_voice import play_wav

TIER_COOLDOWN = {"immediate": 1.2, "near": 2.0, "medium": 3.5, "far": 5.0}
CLIP_DIR = Path(__file__).resolve().parent / "audio" / "egpws"


class PiAudioAlerts:
    def __init__(self, enabled=True, voice=True, speech_rate=280):
        self.enabled = enabled
        self.voice = voice
        self.muted = False
        self._lock = threading.Lock()
        self._last_key = ""
        self._last_at = 0.0
        self._queue: list[DeerThreat] = []
        self._busy = threading.Event()
        self._pygame_ok = self._init_pygame()
        self._worker = threading.Thread(target=self._run, name="pi-audio", daemon=True)
        self._worker.start()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.pre_init(frequency=11025, size=-16, channels=1, buffer=256)
            pygame.mixer.init()
            return True
        except Exception:
            return False

    def _run(self):
        while True:
            if not self._queue:
                time.sleep(0.02)
                continue
            threat = self._queue.pop(0)
            self._busy.set()
            try:
                self._play_alert(threat)
            finally:
                if not self._queue:
                    self._busy.clear()

    def wait_idle(self, timeout: float | None = None) -> bool:
        if not self._busy.is_set() and not self._queue:
            return True
        if timeout is None:
            while self._busy.is_set() or self._queue:
                time.sleep(0.05)
            return True
        deadline = time.time() + timeout
        while self._busy.is_set() or self._queue:
            if time.time() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def play_sync(self, threat: DeerThreat) -> None:
        if not self.enabled or self.muted:
            return
        self._play_alert(threat)

    def _deer_clip(self) -> Path | None:
        for name in ("deer_deer.wav", "deer.wav"):
            p = CLIP_DIR / name
            if p.exists():
                return p
        return None

    def _play_alert(self, threat: DeerThreat) -> None:
        if not self.voice:
            print("[ALERT] Deer, deer")
            return
        clip = self._deer_clip()
        if clip:
            play_wav(clip, pygame_ok=self._pygame_ok)
            return
        from agm_egpws_voice import EGPWSVoice
        EGPWSVoice(enabled=True).speak_deer()

    def trigger_threat(self, threat: DeerThreat) -> None:
        if not self.enabled or self.muted:
            return
        key = f"{threat.tier}:{threat.side}"
        now = time.time()
        cd = TIER_COOLDOWN.get(threat.tier, 3.0)
        with self._lock:
            if key == self._last_key and now - self._last_at < cd:
                return
            self._last_key = key
            self._last_at = now
        self._queue.append(threat)