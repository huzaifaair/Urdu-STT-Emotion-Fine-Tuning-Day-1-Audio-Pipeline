"""
Stage 3 (optional): speaker diarization via pyannote.audio.

This stage is OPTIONAL and degrades gracefully:

- If `USE_DIARIZATION` is not enabled, or pyannote/your HF token is unavailable, diarization
  is skipped. In that case the pipeline tags every segment `SPEAKER_0` and marks
  `crosstalk_flag=False` with `crosstalk_method="none"` (i.e. crosstalk was NOT evaluated).
  This is logged loudly so you know to re-run with diarization before trusting speaker/crosstalk.

- When enabled (set `USE_DIARIZATION=1` and `HUGGINGFACE_TOKEN=...`), pyannote's
  speaker-diarization-3.1 pipeline runs on the cleaned source file and returns speaker turns.
  Segment speaker IDs and crosstalk flags are then derived from those turns (see segment.py).

Why optional? pyannote requires a HuggingFace token + a one-time model download, which is
incompatible with the eventual air-gapped deployment. You must run it *before* air-gapping.
The rest of the pipeline (denoise/segment/manifest/transcribe) does not need it.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from common import (USE_DIARIZATION, HF_TOKEN, DIARIZATION_MODEL, TARGET_SR, logger)


def diarize_file(wav_path: str | Path) -> tuple[list[tuple[float, float, str]], str]:
    """
    Run diarization on a 16 kHz mono WAV file.

    Returns (turns, status) where:
      turns = list of (start_s, end_s, speaker_label) e.g. ("SPEAKER_1", ...)
      status = "ok" | "skipped" | "error"

    If diarization is unavailable, returns ([], "skipped").
    """
    if not USE_DIARIZATION:
        logger.info("Diarization DISABLED (USE_DIARIZATION not set). "
                    "Tagging all segments SPEAKER_0; crosstalk NOT evaluated.")
        return [], "skipped"

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        logger.warning("pyannote.audio not installed -> diarization skipped (%s).", e)
        return [], "skipped"

    if not HF_TOKEN:
        logger.warning("USE_DIARIZATION=1 but no HUGGINGFACE_TOKEN set -> diarization skipped.")
        return [], "skipped"

    try:
        logger.info("Loading diarization pipeline %s ...", DIARIZATION_MODEL)
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
        logger.info("Running diarization on %s ...", Path(wav_path).name)
        diarization = pipeline(str(wav_path))

        turns: list[tuple[float, float, str]] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            label = speaker if speaker.startswith("SPEAKER_") else f"SPEAKER_{speaker}"
            turns.append((float(segment.start), float(segment.end), label))
        logger.info("Diarization done: %d speaker turns.", len(turns))
        return turns, "ok"
    except Exception as e:
        logger.error("Diarization failed for %s: %s", wav_path, e)
        return [], "error"
