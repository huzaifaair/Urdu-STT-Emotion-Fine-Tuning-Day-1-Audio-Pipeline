"""
Shared configuration, paths, and logging helpers for the Day-1 Urdu STT audio pipeline.

All paths are resolved relative to the workspace root (the folder that contains this
`pipeline` directory). Output is written to <workspace>/processed/...
"""

from __future__ import annotations

import os
import sys
import json
import logging
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace / output layout
# ---------------------------------------------------------------------------
# This file lives at <workspace>/pipeline/common.py, so the workspace root is its parent.
WORKSPACE = Path(__file__).resolve().parent.parent

PROCESSED_DIR = WORKSPACE / "processed"
CLEAN_DIR = PROCESSED_DIR / "clean"
SEGMENTS_DIR = PROCESSED_DIR / "segments"
MANIFEST_PATH = PROCESSED_DIR / "manifest.jsonl"
ERRORS_LOG = PROCESSED_DIR / "errors.log"
TRANSCRIPT_LOG = PROCESSED_DIR / "transcription.log"

# ---------------------------------------------------------------------------
# Audio processing parameters
# ---------------------------------------------------------------------------
TARGET_SR = 16000          # Whisper + pyannote both expect 16 kHz mono
TARGET_CHANNELS = 1

# Segmentation (seconds)
SEGMENT_TARGET = 20.0      # nominal segment length; kept within 15-30s window
SEGMENT_MIN = 15.0         # lower bound of the desired trainable range
SEGMENT_MAX = 30.0         # upper bound of the desired trainable range
TOO_SHORT = 3.0            # anything shorter than this is "too short to be trainable"
BOUNDARY_SNAP = 2.0        # snap a segment boundary to nearest silence within +- this many seconds

# Denoising / normalization
LUFS_TARGET = -23.0        # EBU R128 integrated loudness target
DENOISE_PROP_DECREASE = 0.9
DENOISE_STATIONARY = True  # stationary=True is much faster; set False for heavier non-stationary reduction
DENOISE_CHUNK_S = 30.0     # chunk length (s) for memory-bounded non-stationary reduction

# Dataset / manifest
DATASET_VERSION = "v1"

# ---------------------------------------------------------------------------
# Day-2 labeling / review / split
# ---------------------------------------------------------------------------
REVIEW_DIR = PROCESSED_DIR / "review"
REVIEW_HTML = REVIEW_DIR / "review.html"
REVIEW_LABELS = REVIEW_DIR / "review_labels.json"

SPLITS_DIR = PROCESSED_DIR / "splits"

# Manual-label vocabularies. "unknown" is the explicit "not yet labeled" sentinel.
ACCENTS = ["unknown", "karachi", "lahori", "punjabi_influenced",
           "sindhi_influenced", "pashtun_influenced", "other"]
EMOTIONS = ["neutral", "happy", "angry", "sad"]
EMOTION_UNKNOWN = "unknown"

# Train/val/test proportions and the RNG seed used for the deterministic split.
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

# Transcription
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # multilingual "base" handles Urdu

# Diarization (optional). Requires a HuggingFace token + one-time model download.
# Set HUGGINGFACE_TOKEN and USE_DIARIZATION=1 to enable. When disabled, every segment is
# tagged SPEAKER_0 and crosstalk is reported as not-evaluated (see diarize.py notes).
USE_DIARIZATION = os.environ.get("USE_DIARIZATION", "0") == "1"
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
# pyannote pipeline identifier (separation + diarization). Swap if you prefer a newer/free model.
DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
)

# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------
# The brief referenced /podcast and /fresh_recording, but the workspace actually contains
# "Podcasts" and "Fresh_rec" (different casing/spelling). We auto-detect by trying several
# candidate names so the pipeline "just works" regardless of which spelling exists, and so
# that newly added folders are picked up automatically. Override with env vars if needed:
#   PODCAST_DIR, FRESH_RECORDING_DIR
SOURCE_CANDIDATES = {
    "podcast": ["podcast", "Podcasts", "Podcast", "PODCAST"],
    "fresh_recording": ["fresh_recording", "fresh_rec", "Fresh_rec",
                        "Fresh Recording", "Fresh_Recording", "fresh recordings"],
}


def resolve_source_dir(source_type: str) -> Path | None:
    """Return the first existing directory for a source type, or None if absent/empty-handled."""
    env_override = {
        "podcast": os.environ.get("PODCAST_DIR"),
        "fresh_recording": os.environ.get("FRESH_RECORDING_DIR"),
    }.get(source_type)
    candidates = []
    if env_override:
        candidates.append(env_override)
    candidates.extend(SOURCE_CANDIDATES[source_type])

    for name in candidates:
        p = (WORKSPACE / name)
        if p.is_dir():
            return p
    return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Module-level logger; handlers are attached by setup_logging() (idempotent).
logger = logging.getLogger("day1_pipeline")


def setup_logging() -> logging.Logger:
    """Configure a root logger that writes to stderr with timestamps and stage tags."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Also mirror INFO+ to a run log inside processed/ for later review.
    fh = logging.FileHandler(PROCESSED_DIR / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger._configured = True
    return logger


def log_error(logger: logging.Logger, stage: str, item: str, exc: Exception) -> None:
    """Append a structured error line to errors.log and log a short message. Never raises."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    line = f"{_now()} | STAGE={stage} | ITEM={item}\n{tb}\n{'-'*80}\n"
    try:
        with open(ERRORS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    logger.error("FAILED stage=%s item=%s : %s", stage, item, exc)


def sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_manifest_line(record: dict) -> None:
    """Append one JSON object (UTF-8) as a line to the manifest."""
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """Load all records from a JSONL manifest. Returns [] if the file is absent."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_manifest(records: list[dict], path: Path = MANIFEST_PATH) -> None:
    """Atomically rewrite a JSONL manifest (UTF-8, one record per line)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
