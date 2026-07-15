"""
Day-1 pipeline orchestrator (Stages 1-4):
  1. Denoise + EBU R128 loudness-normalize each source file  -> processed/clean/
  2. Segment into 15-30s clips (flag too-short + crosstalk)  -> processed/segments/
  3. Speaker diarization (optional) tags each segment        (pyannote)
  4. Write manifest.jsonl (UTF-8, one record per segment)

Stage 5 (background Whisper transcription) is launched separately by transcribe.py, or by
passing --transcribe to this script (it spawns a detached background process).

Usage:
  python pipeline/day1_pipeline.py                 # full run
  python pipeline/day1_pipeline.py --limit-seconds 60   # process only first 60s of each file (smoke test)
  python pipeline/day1_pipeline.py --transcribe    # also kick off transcription in background when done

Errors are caught per-file / per-segment, written to processed/errors.log, and the run continues.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# allow running as `python pipeline/day1_pipeline.py` from the workspace root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WORKSPACE, PROCESSED_DIR, CLEAN_DIR, SEGMENTS_DIR, MANIFEST_PATH,
    DATASET_VERSION, TARGET_SR, resolve_source_dir, setup_logging, log_error,
    sha256_of_file, logger, USE_DIARIZATION,
)
from audio_io import load_audio, save_audio  # noqa: E402
from denoise import clean_audio  # noqa: E402
from diarize import diarize_file  # noqa: E402
from segment import segment_audio  # noqa: E402

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg",".mpeg")


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def process_source(source_type: str, src_dir: Path, limit_seconds: float | None,
                   stats: dict) -> None:
    files = sorted([p for p in src_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    if not files:
        logger.info("[%s] no audio files found in %s", source_type, src_dir)
        return

    clean_sub = CLEAN_DIR / source_type
    seg_sub = SEGMENTS_DIR / source_type
    clean_sub.mkdir(parents=True, exist_ok=True)
    seg_sub.mkdir(parents=True, exist_ok=True)

    for f in files:
        logger.info("=" * 70)
        logger.info("[%s] STAGE 1/2/3: processing %s", source_type, f.name)
        try:
            waveform, sr = load_audio(f, TARGET_SR)
        except Exception as e:
            log_error(logger, "load", str(f), e)
            stats["load_errors"] += 1
            continue

        if limit_seconds:
            waveform = waveform[: int(limit_seconds * sr)]
            logger.info("  (limit-seconds=%s -> using first %.1fs, %d samples)",
                        limit_seconds, limit_seconds, waveform.size)

        # ---- Stage 1: denoise + normalize ----
        try:
            cleaned = clean_audio(waveform, sr)
        except Exception as e:
            log_error(logger, "denoise", str(f), e)
            stats["denoise_errors"] += 1
            continue

        clean_name = f.stem + ".wav"
        clean_path = clean_sub / clean_name
        try:
            save_audio(clean_path, cleaned, sr)
        except Exception as e:
            log_error(logger, "save_clean", str(f), e)
            stats["save_errors"] += 1
            continue
        logger.info("  saved cleaned audio -> %s", clean_path)

        # ---- Stage 3: diarization (optional) on the cleaned file ----
        try:
            turns, dia_status = diarize_file(clean_path)
        except Exception as e:
            log_error(logger, "diarize", str(f), e)
            turns, dia_status = [], "error"
            stats["diarize_errors"] += 1

        # ---- Stage 2: segmentation (+ speaker/crosstalk tagging) ----
        try:
            segments = segment_audio(cleaned, sr, turns, f.stem, dia_status)
        except Exception as e:
            log_error(logger, "segment", str(f), e)
            stats["segment_errors"] += 1
            continue

        if not segments:
            logger.warning("  no segments produced for %s", f.name)
            continue

        # ---- Stage 4: save segments + write manifest records ----
        stem = _sanitize(f.stem)
        for i, seg in enumerate(segments):
            seg_num = f"{i:04d}"
            start_s = seg["start"]
            end_s = seg["end"]
            fname = f"{stem}_{seg_num}_{start_s:07.2f}_{end_s:07.2f}.wav"
            seg_path = seg_sub / fname
            try:
                save_audio(seg_path, seg["waveform"], sr)
            except Exception as e:
                log_error(logger, "save_segment", fname, e)
                stats["save_errors"] += 1
                continue

            try:
                checksum = sha256_of_file(seg_path)
            except Exception as e:
                log_error(logger, "checksum", fname, e)
                checksum = ""

            record = {
                "segment_id": seg_path.stem,
                "filename": fname,
                "source_file": f.name,
                "source_path": str(f.resolve()),
                "source_type": source_type,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration_seconds": seg["duration"],
                "speaker_id": seg["speaker_id"],
                "checksum_sha256": checksum,
                "dataset_version": DATASET_VERSION,
                "crosstalk_flag": seg["crosstalk_flag"],
                "crosstalk_method": seg["crosstalk_method"],
                "too_short": seg["too_short"],
                "verified": False,
                "transcript": None,
            }
            try:
                with open(MANIFEST_PATH, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                log_error(logger, "manifest_write", fname, e)
                stats["manifest_errors"] += 1
                continue

            stats["segments"] += 1
            if seg["too_short"]:
                stats["too_short"] += 1
            if seg["crosstalk_flag"]:
                stats["crosstalk"] += 1

        logger.info("  done: %d segments written for %s", len(segments), f.name)


def maybe_launch_transcription() -> None:
    """Spawn transcribe.py as a detached background process the user can monitor."""
    script = Path(__file__).resolve().parent / "transcribe.py"
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        popen_kwargs = dict(creationflags=creationflags, close_fds=True)
    else:
        popen_kwargs = dict(start_new_session=True)
    try:
        subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(WORKSPACE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs,
        )
        logger.info("Kicked off background transcription job: python pipeline/transcribe.py")
        logger.info("  - tail progress with:  type processed\\transcription.log (Windows) "
                    "or  tail -f processed/transcription.log")
    except Exception as e:
        log_error(logger, "launch_transcribe", "transcribe.py", e)
        logger.error("Could not auto-launch transcription; run `python pipeline/transcribe.py` manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-1 Urdu STT audio pipeline (stages 1-4).")
    parser.add_argument("--limit-seconds", type=float, default=None,
                        help="Process only the first N seconds of each source file (smoke test).")
    parser.add_argument("--transcribe", action="store_true",
                        help="Kick off the background Whisper transcription job when done.")
    args = parser.parse_args()

    setup_logging()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh manifest each run (truncate) so re-runs don't duplicate-append.
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        logger.info("Removed previous manifest to avoid duplicate appends.")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Workspace: %s", WORKSPACE)
    logger.info("Diarization: %s", "ENABLED" if USE_DIARIZATION else "disabled (SPEAKER_0 + crosstalk not evaluated)")
    logger.info("Outputs -> %s", PROCESSED_DIR)

    stats = {
        "segments": 0, "too_short": 0, "crosstalk": 0, "transcribed": 0,
        "load_errors": 0, "denoise_errors": 0, "segment_errors": 0,
        "diarize_errors": 0, "save_errors": 0, "manifest_errors": 0,
    }

    for source_type in ("podcast", "fresh_recording"):
        src_dir = resolve_source_dir(source_type)
        if src_dir is None:
            logger.info("[%s] source directory not found -> skipped silently.", source_type)
            continue
        logger.info("[%s] using source directory: %s", source_type, src_dir)
        process_source(source_type, src_dir, args.limit_seconds, stats)

    print_summary(stats)

    if args.transcribe:
        maybe_launch_transcription()


def print_summary(stats: dict) -> None:
    line = "=" * 70
    logger.info(line)
    logger.info("DAY-1 PIPELINE SUMMARY")
    logger.info("  total segments created : %d", stats["segments"])
    logger.info("  flagged too short      : %d", stats["too_short"])
    logger.info("  flagged crosstalk       : %d", stats["crosstalk"])
    logger.info("  transcribed (so far)    : %d", stats["transcribed"])
    logger.info("  failures:")
    logger.info("    load errors           : %d", stats["load_errors"])
    logger.info("    denoise errors        : %d", stats["denoise_errors"])
    logger.info("    segment errors        : %d", stats["segment_errors"])
    logger.info("    diarize errors        : %d", stats["diarize_errors"])
    logger.info("    save errors           : %d", stats["save_errors"])
    logger.info("    manifest errors       : %d", stats["manifest_errors"])
    logger.info("  manifest: %s", MANIFEST_PATH)
    logger.info("  errors log: %s", PROCESSED_DIR / "errors.log")
    logger.info(line)


if __name__ == "__main__":
    main()
