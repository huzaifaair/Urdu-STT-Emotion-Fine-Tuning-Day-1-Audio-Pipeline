"""
Stage 5 (background/overnight): draft transcription of every segment with Whisper.

This is meant to run as a long-lived background job. It:
  - reads processed/manifest.jsonl,
  - loads Whisper (default "base", multilingual -> handles Urdu),
  - transcribes each segment WAV whose `transcript` is still null,
  - writes the draft text back into manifest.jsonl (matched by segment_id),
  - writes progress lines like "150/400 segments transcribed (37.5%)" to
    processed/transcription.log (safe to `tail` / `type` while it runs),
  - logs per-segment failures to processed/errors.log and keeps going.

Run it:
  python pipeline/transcribe.py                 # resume: skip already-transcribed
  python pipeline/transcribe.py --force         # re-transcribe everything
  python pipeline/transcribe.py --model small   # use a bigger model (slower)

It is intentionally separate from day1_pipeline.py so transcription can run unattended
overnight while you inspect/manually correct earlier segments.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WORKSPACE, PROCESSED_DIR, SEGMENTS_DIR, MANIFEST_PATH, TRANSCRIPT_LOG,
    WHISPER_MODEL, TARGET_SR, setup_logging, log_error, logger,
)

# Whisper needs the ffmpeg CLI on PATH for some operations; point it at the bundled binary.
import imageio_ffmpeg  # noqa: E402
try:
    _ff = imageio_ffmpeg.get_ffmpeg_exe()
    _ffdir = str(Path(_ff).parent)
    import os
    os.environ["PATH"] = _ffdir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = _ff
except Exception:
    pass


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_manifest(records):
    tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST_PATH)


def _progress(done: int, total: int) -> str:
    pct = (100.0 * done / total) if total else 0.0
    return f"{done}/{total} segments transcribed ({pct:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Whisper draft transcription (background).")
    parser.add_argument("--model", default=WHISPER_MODEL, help="Whisper model name (base/small/medium).")
    parser.add_argument("--force", action="store_true", help="Re-transcribe segments that already have a transcript.")
    parser.add_argument("--batch", type=int, default=20, help="Rewrite manifest every N segments.")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting transcription job (model=%s, force=%s)", args.model, args.force)

    import whisper  # imported here so stage 1-4 can run even if whisper were unavailable

    records = load_manifest()
    if not records:
        logger.error("No manifest records found at %s. Run day1_pipeline.py first.", MANIFEST_PATH)
        return

    pending = [r for r in records if args.force or r.get("transcript") in (None, "")]
    total = len(pending)
    logger.info("Segments total=%d, to transcribe=%d", len(records), total)
    if total == 0:
        logger.info("Nothing to transcribe.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_LOG, "w", encoding="utf-8") as lf:
        lf.write(f"[{_ts()}] Transcription started. {_progress(0, total)}\n")

    logger.info("Loading Whisper model '%s' ...", args.model)
    model = whisper.load_model(args.model)
    logger.info("Whisper model loaded.")

    done = 0
    start = time.time()
    for r in pending:
        seg_id = r.get("segment_id")
        fname = r.get("filename")
        # locate the wav: try source subfolder first, then any
        wav = SEGMENTS_DIR / r.get("source_type", "") / fname
        if not wav.exists():
            wav = SEGMENTS_DIR / fname
        if not wav.exists():
            log_error(logger, "transcribe", seg_id, FileNotFoundError(f"{fname} not found"))
            done += 1
            continue
        try:
            res = model.transcribe(str(wav), language="ur", task="transcribe",
                                   fp16=False, verbose=False, beam_size=1)
            text = (res.get("text") or "").strip()
            r["transcript"] = text
            r["transcript_model"] = args.model
        except Exception as e:
            log_error(logger, "transcribe", seg_id, e)
            r["transcript"] = r.get("transcript")  # leave as-is (null) on failure

        done += 1
        if done % args.batch == 0 or done == total:
            save_manifest(records)
            msg = _progress(done, total)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as lf:
                lf.write(f"[{_ts()}] {msg}  ({rate:.2f} seg/s, ETA {eta/60:.1f} min)\n")
            logger.info("%s", msg)

    save_manifest(records)
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as lf:
        lf.write(f"[{_ts()}] Transcription finished. {_progress(done, total)}\n")
    logger.info("Transcription complete: %s", _progress(done, total))


def _ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
