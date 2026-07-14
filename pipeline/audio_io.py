"""
Audio I/O helpers.

Decoding is done through the bundled ffmpeg binary (from the `imageio-ffmpeg` package)
piped to raw float32. This deliberately avoids torchcodec/torchaudio's default loader and
system ffmpeg installs, so it works consistently for wav/mp3/m4a on a clean machine.
Encoding uses soundfile (libsndfile) for 16 kHz mono WAV.
"""

from __future__ import annotations

import subprocess
import numpy as np
from pathlib import Path

import imageio_ffmpeg

from common import TARGET_SR, TARGET_CHANNELS, logger  # noqa: F401  (logger injected by caller)


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Decode any supported audio file to a mono float32 waveform at `target_sr`.

    Returns (waveform_1d_float32, sample_rate).
    """
    ff = _ffmpeg_exe()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path),
        "-ac", str(TARGET_CHANNELS), "-ar", str(target_sr),
        "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: {proc.stderr.decode('utf-8', 'ignore')[:500]}"
        )
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    # guard against the rare all-zero/empty decode
    if data.size == 0:
        raise RuntimeError(f"ffmpeg produced empty audio for {path}")
    return data.astype(np.float32), int(target_sr)


def save_audio(path: str | Path, waveform: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a mono float32 waveform to a 16-bit PCM WAV file (UTF-8-safe path)."""
    import soundfile as sf
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(waveform, dtype=np.float32)
    # soft guard: clip to [-1, 1] to avoid WAV overflow artefacts
    w = np.clip(w, -1.0, 1.0)
    sf.write(str(p), w, sr, subtype="PCM_16")
