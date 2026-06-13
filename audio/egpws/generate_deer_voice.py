"""Build harsh MK-VIII-style deer_deer.wav from user reference."""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy.signal import find_peaks, lfilter

ROOT = Path(__file__).resolve().parent
USER_M4A = Path(r"c:\Users\User\Documents\Sound recordings\Recording (autosaved).m4a")
TERRAIN = ROOT / "terrain.wav"
OUT = ROOT / "deer_deer.wav"
TMP = ROOT / "user_reference.wav"
TARGET_SR = 11025
HARSHNESS = 3.0


def ffmpeg_to_wav(src: Path, dst: Path, sr: int = TARGET_SR) -> None:
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [exe, "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return sr, x


def trim_speech(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * 0.01)
    env = np.array([np.sqrt(np.mean(x[i : i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    if env.max() < 1e-6:
        return x
    active = np.where(env > env.max() * 0.15)[0]
    if active.size == 0:
        return x
    start = max(0, (int(active[0]) - 1) * hop)
    end = min(len(x), (int(active[-1]) + 2) * hop)
    return x[start:end]


def emphasize_dee_onsets(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * 0.005)
    env = np.array([np.max(np.abs(x[i : i + hop])) for i in range(0, len(x) - hop, hop)])
    peaks, _ = find_peaks(env, height=env.max() * 0.28, distance=max(1, int(0.06 / 0.005)))
    y = x.copy()
    plosive_gain = 1.6 * HARSHNESS
    for p in peaks:
        i0 = max(0, int(p * hop) - int(0.02 * sr))
        i1 = min(len(y), i0 + int(0.07 * sr))
        seg = y[i0:i1]
        if len(seg) < 8:
            continue
        n = np.arange(len(seg))
        burst = np.exp(-n / (0.008 * sr))
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        gain = np.ones(len(spec))
        gain[(freqs >= 80) & (freqs <= 1200)] *= plosive_gain
        gain[(freqs > 1200) & (freqs <= 4000)] *= plosive_gain * 1.4
        seg2 = np.fft.irfft(spec * gain, len(seg))
        y[i0:i1] = np.clip(seg + seg2 * burst * 0.85, -1.0, 1.0)
    return np.clip(y, -1.0, 1.0)


def _distort(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def harshify_like_terrain(x: np.ndarray, sr: int, terrain: np.ndarray) -> np.ndarray:
    h = HARSHNESS
    pre = min(0.995, 0.92 + 0.02 * h)
    x = lfilter([1.0, -pre], [1.0], x)
    x = _distort(x, 5.5 * h)
    n = len(x)
    spec = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[(freqs > 800) & (freqs < 4500)] *= 2.2 * h
    spec[(freqs > 4500) & (freqs < 7000)] *= 1.5 * h
    x = np.fft.irfft(spec, n)
    comp = _distort(x, 3.0 * h)
    x = np.clip(0.15 * x + 0.85 * comp, -1.0, 1.0)
    x = _distort(x, 4.0 * h)
    x = np.sign(x) * np.minimum(np.abs(x) ** 0.75, 1.0)
    return np.clip(x / (np.max(np.abs(x)) + 1e-9), -1.0, 1.0)


def maximize_loudness(x: np.ndarray) -> np.ndarray:
    """Normalize to full digital scale - as loud as the format allows."""
    x = np.clip(x, -1.0, 1.0)
    peak = float(np.max(np.abs(x)) + 1e-9)
    x = x / peak
    x = np.sign(x) * np.minimum(np.abs(x) ** 0.72, 1.0)
    peak = float(np.max(np.abs(x)) + 1e-9)
    return np.clip(x / peak, -1.0, 1.0)


def write_8bit(path: Path, sr: int, x: np.ndarray) -> None:
    x = np.clip(x, -1.0, 1.0)
    out = np.clip(np.floor(x * 127 + 128), 0, 255).astype(np.uint8)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(sr)
        w.writeframes(out.tobytes())


def main() -> int:
    if not USER_M4A.exists() or not TERRAIN.exists():
        print("Missing reference files")
        return 1
    ffmpeg_to_wav(USER_M4A, TMP)
    sr, x = load_wav(TMP)
    _, terrain = load_wav(TERRAIN)
    x = trim_speech(x, sr)
    x = emphasize_dee_onsets(x, sr)
    x = harshify_like_terrain(x, sr, terrain)
    x = maximize_loudness(x)
    write_8bit(OUT, sr, x)
    sp = x[int(0.03 * sr) : int(0.95 * len(x))]
    spec = np.abs(np.fft.rfft(sp * np.hanning(len(sp))))
    freqs = np.fft.rfftfreq(len(sp), 1 / sr)
    cent = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
    print(
        f"wrote {OUT} | harshness={HARSHNESS}x | rms {np.sqrt(np.mean(x**2)):.3f} | "
        f"peak {np.max(np.abs(x)):.3f} | centroid {cent:.0f} Hz"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())