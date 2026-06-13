"""EGPWS voice playback helpers."""
from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLIP_DIR = ROOT / "audio" / "egpws"

PLAYBACK_GAIN = 5.0
EGPWS_WPM = 280
ESPEAK_VOICE = "en-us+m3"
WINDOWS_SAPI_RATE = 4
PREFERRED_WINDOWS_VOICES = ("david", "mark", "george", "richard")


def _load_wav_float(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return sr, x


def play_wav(path: Path, pygame_ok: bool = True, gain: float = PLAYBACK_GAIN) -> None:
    sr, x = _load_wav_float(path)
    x = np.clip(x * gain, -1.0, 1.0)
    if pygame_ok:
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(frequency=sr, size=-16, channels=1, buffer=256)
                pygame.mixer.init()
            audio = (x * 32767).astype(np.int16)
            snd = pygame.sndarray.make_sound(audio)
            snd.set_volume(1.0)
            ch = snd.play()
            while ch.get_busy():
                __import__("time").sleep(0.005)
            return
        except Exception:
            pass
    if platform.system() == "Windows":
        try:
            import winsound
            import tempfile
            tmp = Path(tempfile.gettempdir()) / "_egpws_boost.wav"
            out = np.clip(x * 127 + 128, 0, 255).astype(np.uint8)
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(1)
                w.setframerate(sr)
                w.writeframes(out.tobytes())
            winsound.PlaySound(str(tmp), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            return
        except Exception:
            pass
    try:
        subprocess.run(["aplay", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class EGPWSVoice:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._engine = None
        self._pygame_ok = False
        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import pygame
            pygame.mixer.pre_init(frequency=11025, size=-16, channels=1, buffer=256)
            pygame.mixer.init()
            self._pygame_ok = True
        except Exception:
            pass
        if enabled and platform.system() == "Windows":
            self._init_windows_engine()

    def speak_deer(self) -> None:
        for name in ("deer_deer.wav", "deer.wav"):
            clip = CLIP_DIR / name
            if clip.exists():
                play_wav(clip, pygame_ok=self._pygame_ok)
                return
        with self._lock:
            if platform.system() == "Windows":
                self._speak_windows("deer deer")
            else:
                self._speak_espeak("deer deer")

    def _init_windows_engine(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", EGPWS_WPM)
            engine.setProperty("volume", 1.0)
            for voice in engine.getProperty("voices"):
                name = (voice.name or "").lower()
                if any(v in name for v in PREFERRED_WINDOWS_VOICES):
                    engine.setProperty("voice", voice.id)
                    break
            self._engine = engine
        except Exception:
            self._engine = None

    def _speak_windows(self, text: str) -> None:
        if self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception:
                pass
        safe = text.replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {WINDOWS_SAPI_RATE}; $s.Volume = 100; "
            "foreach ($name in @('Microsoft David Desktop','Microsoft Mark Desktop')) { "
            "  try { $s.SelectVoice($name); break } catch {} "
            "} "
            f"$s.Speak('{safe}')"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=4, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _espeak_bin(self) -> str | None:
        for name in ("espeak-ng", "espeak"):
            p = shutil.which(name)
            if p:
                return p
        return None

    def _speak_espeak(self, text: str) -> None:
        exe = self._espeak_bin()
        if not exe:
            print(f"[EGPWS] {text}")
            return
        subprocess.run([exe, "-v", ESPEAK_VOICE, "-s", str(EGPWS_WPM), "-p", "40", "-a", "255", text], check=False, timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)