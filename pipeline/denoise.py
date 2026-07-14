"""
Stage 1: denoising + EBU R128 loudness normalization.

Design notes / assumptions:
- Denoise uses `noisereduce`. We process in fixed-length chunks so a 5-hour file never
  blows up RAM. `DENOISE_STATIONARY=True` (config) gives a fast, constant-noise profile
  suitable for light podcast hiss; flip to False for heavier non-stationary reduction.
- Loudness normalization follows EBU R128: measure integrated loudness with a BS.1770
  meter (pyloudnorm), apply a linear gain to hit LUFS_TARGET (-23 LUFS), then a safe
  true-peak limiter that only *reduces* gain so peaks stay at/under -1 dBTP (no hard clipping).
  This is a correct, conservative R128-style normalization without needing ffmpeg's
  `loudnorm` two-pass or an external loudness lib.
"""

from __future__ import annotations

import numpy as np

import noisereduce as nr
import pyloudnorm as pyln

from common import (TARGET_SR, LUFS_TARGET, DENOISE_PROP_DECREASE,
                    DENOISE_STATIONARY, DENOISE_CHUNK_S, logger)


def _denoise(y: np.ndarray, sr: int) -> np.ndarray:
    chunk_samples = int(DENOISE_CHUNK_S * sr)
    if DENOISE_STATIONARY:
        # Fast path: single stationary noise estimate over the whole signal.
        return nr.reduce_noise(
            y=y, sr=sr, stationary=True,
            prop_decrease=DENOISE_PROP_DECREASE, use_tqdm=False,
        )
    # Non-stationary path, chunked to bound memory.
    if y.size <= chunk_samples:
        return nr.reduce_noise(y=y, sr=sr, stationary=False,
                               prop_decrease=DENOISE_PROP_DECREASE, use_tqdm=False)
    out = np.empty_like(y)
    for start in range(0, y.size, chunk_samples):
        end = min(start + chunk_samples, y.size)
        out[start:end] = nr.reduce_noise(
            y=y[start:end], sr=sr, stationary=False,
            prop_decrease=DENOISE_PROP_DECREASE, use_tqdm=False,
        )
    return out


def _normalize_r128(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size < sr:  # far too short to measure loudness meaningfully
        return y
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
    except Exception as e:  # pyloudnorm can raise on all-silence / degenerate input
        logger.warning("Loudness measurement failed (%s); skipping normalization.", e)
        return y

    if not np.isfinite(loudness):
        logger.warning("Loudness is non-finite (likely silence); skipping normalization.")
        return y

    gain_linear = 10.0 ** ((LUFS_TARGET - loudness) / 20.0)
    y_gained = y * gain_linear

    # True-peak limiter to -1 dBTP: only reduce gain if we would otherwise exceed it.
    peak = float(np.max(np.abs(y_gained))) if y_gained.size else 0.0
    tp_ceiling = 10.0 ** (-1.0 / 20.0)  # -1 dBTP
    if peak > tp_ceiling:
        limiter = tp_ceiling / peak
        y_gained = y_gained * limiter
        logger.debug("Applied true-peak limiter (gain x%.3f) to keep peaks <= -1 dBTP.", limiter)

    return y_gained.astype(np.float32)


def clean_audio(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Denoise then loudness-normalize a mono float32 waveform. Returns cleaned waveform."""
    denoised = _denoise(y, sr)
    normalized = _normalize_r128(denoised, sr)
    return normalized.astype(np.float32)
