"""
Stage 2 (+ speaker/crosstalk tagging from Stage 3): segmentation.

- Splits the cleaned waveform into ~15-30s non-overlapping windows (nominal 20s).
- Segment boundaries are snapped to the nearest low-energy point (silence) within a small
  window, to avoid cutting words in half where possible.
- Flags segments shorter than TOO_SHORT (default 3s) as `too_short` (kept, not silently dropped,
  so you can review) -- these are "too short to be trainable".
- Tags each segment with a speaker ID (majority speaker from diarization turns) and a
  crosstalk flag (>=2 distinct speakers overlapping in time within the segment).
  When diarization was skipped, speaker=SPEAKER_0 and crosstalk is reported as not-evaluated.
"""

from __future__ import annotations

import numpy as np

from common import (TARGET_SR, SEGMENT_TARGET, TOO_SHORT, BOUNDARY_SNAP, logger)


def _short_time_energy(y: np.ndarray, sr: int, win_s: float = 0.025) -> np.ndarray:
    win = max(1, int(win_s * sr))
    if y.size < win:
        return np.array([float(np.sum(y ** 2))], dtype=np.float32)
    # simple moving average of frame energy via convolution
    power = y ** 2
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(power, kernel, mode="same")


def _snap_to_silence(energy: np.ndarray, center_idx: int, sr: int, max_shift_s: float) -> int:
    """Return the index nearest `center_idx` (within +-max_shift_s) with minimal energy."""
    max_shift = int(max_shift_s * sr)
    lo = max(0, center_idx - max_shift)
    hi = min(energy.size, center_idx + max_shift + 1)
    if hi <= lo:
        return center_idx
    local = energy[lo:hi]
    shift = int(np.argmin(local)) - (center_idx - lo)
    return center_idx + shift


def _speaker_and_crosstalk(turns, seg_start, seg_end):
    """
    Given diarization turns [(start, end, speaker), ...] and a segment window,
    return (speaker_id, crosstalk_flag).
    Speaker = the one with the most overlapping time; crosstalk = two different speakers
    overlap within the window by more than a small threshold.
    """
    if not turns:
        return "SPEAKER_0", False

    coverage: dict[str, float] = {}
    overlap_pairs = 0.0
    for (s, e, spk) in turns:
        ov_s = max(s, seg_start)
        ov_e = min(e, seg_end)
        ov = max(0.0, ov_e - ov_s)
        if ov <= 0:
            continue
        coverage[spk] = coverage.get(spk, 0.0) + ov

    if not coverage:
        return "SPEAKER_0", False

    speaker_id = max(coverage.items(), key=lambda kv: kv[1])[0]

    # Crosstalk: any two distinct speakers whose intervals overlap inside this window.
    active = [(max(s, seg_start), min(e, seg_end), spk)
              for (s, e, spk) in turns if min(e, seg_end) - max(s, seg_start) > 0]
    crosstalk = False
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if active[i][2] == active[j][2]:
                continue
            ov = min(active[i][1], active[j][1]) - max(active[i][0], active[j][0])
            if ov > 0.3:  # require a meaningful overlap (>300 ms) to avoid edge artifacts
                crosstalk = True
                break
        if crosstalk:
            break

    return speaker_id, crosstalk


def segment_audio(y: np.ndarray, sr: int, turns, source_stem: str,
                  diarization_status: str):
    """
    Segment a cleaned waveform. Returns a list of dicts with keys:
      start, end, duration, waveform, speaker_id, too_short, crosstalk_flag, crosstalk_method
    """
    duration = y.size / sr
    energy = _short_time_energy(y, sr)

    segments = []
    pos = 0.0
    seg_idx = 0
    while pos < duration - 1e-3:
        nominal_end = min(pos + SEGMENT_TARGET, duration)
        # snap the end boundary to nearby silence
        end_idx = _snap_to_silence(energy, int(nominal_end * sr), sr, BOUNDARY_SNAP)
        end = end_idx / sr
        start = pos
        if end - start < 0.5:  # degenerate sliver after snapping; extend to nominal_end
            end = nominal_end
        if end <= start:
            end = duration

        speaker_id, crosstalk = _speaker_and_crosstalk(turns, start, end)
        too_short = (end - start) < TOO_SHORT
        crosstalk_method = "diarization" if diarization_status == "ok" else "none"

        seg_wave = y[int(round(start * sr)):int(round(end * sr))].copy()
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "waveform": seg_wave,
            "speaker_id": speaker_id,
            "too_short": too_short,
            "crosstalk_flag": crosstalk,
            "crosstalk_method": crosstalk_method,
        })

        seg_idx += 1
        pos = end
        if end >= duration - 1e-3:
            break

    n_too_short = sum(1 for s in segments if s["too_short"])
    n_crosstalk = sum(1 for s in segments if s["crosstalk_flag"])
    logger.info("Segmented %s -> %d segments (%d too_short, %d crosstalk).",
                source_stem, len(segments), n_too_short, n_crosstalk)
    return segments
