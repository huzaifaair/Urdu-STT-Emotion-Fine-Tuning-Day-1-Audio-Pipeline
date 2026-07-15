"""
Stage 5 (background/overnight): draft transcription of segments.

DEFAULT ENGINE (backend="hf"):
  Hugging Face Urdu-specific checkpoint "ihanif/whisper-medium-urdu" via transformers.
  Audio is loaded with librosa at 16 kHz mono (defensive resample in case a segment
  isn't already 16 kHz), feature-extracted with WhisperProcessor, and run through
  model.generate() directly with explicit forced_decoder_ids (language="urdu",
  task="transcribe"). The ASR pipeline wrapper is intentionally NOT used: this 2023
  fine-tune ships an outdated generation_config, and the pipeline's generate_kwargs
  plumbing surfaces a "model_kwargs are not used" error for forced_decoder_ids.
  Calling generate() directly is the workaround from huggingface/transformers#25084.

LEGACY ENGINE (backend="whisper", env TRANSCRIPTION_BACKEND=whisper):
  The old openai-whisper "base" path, kept for fallback/comparison.

Segment selection (which records get (re)transcribed):
  - ALWAYS skip records where verified == True (manually corrected; never overwrite).
  - By default, only (re)process records whose transcript is empty/None OR whose draft
    looks like the old low-quality base output = a mix of Latin AND Urdu characters in the
    same string (heuristic matches both [A-Za-z] and [\\u0600-\\u06FF]).
  - With --force, process every non-verified record (verified is still protected).

Outputs preserved from before:
  - writes the draft text back into manifest.jsonl (matched by segment_id),
  - updates transcript_model to the id actually used,
  - batch-saves every N records (atomic write via save_manifest),
  - writes progress lines "150/400 segments transcribed (37.5%)" to processed/transcription.log,
  - logs per-segment failures to processed/errors.log and keeps going,
  - prints a run summary (verified-skipped / reprocessed / unchanged / errors).

Run it:
  python pipeline/transcribe.py                 # re-transcribe unverified + low-quality (HF Urdu)
  python pipeline/transcribe.py --force         # re-transcribe all non-verified
  TRANSCRIPTION_BACKEND=whisper python pipeline/transcribe.py   # legacy openai-whisper

It is intentionally separate from day1_pipeline.py so transcription can run unattended
overnight while you inspect/manually correct earlier segments.
"""

from __future__ import annotations

import sys
import re
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WORKSPACE, PROCESSED_DIR, SEGMENTS_DIR, MANIFEST_PATH, TRANSCRIPT_LOG,
    WHISPER_MODEL, TARGET_SR, TRANSCRIPTION_BACKEND, HF_ASR_MODEL_ID,
    setup_logging, log_error, logger,
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

# Mixed-script heuristic: a Latin letter AND an Urdu-range character in the same string.
_LATIN = re.compile(r"[A-Za-z]")
_URDU = re.compile(r"[\u0600-\u06FF]")


def _is_mixed_script(text: str) -> bool:
    return bool(_LATIN.search(text)) and bool(_URDU.search(text))


def _needs_retranscript(record: dict) -> bool:
    """A record needs (re)transcription if its draft is empty or looks like low-quality base output."""
    t = record.get("transcript")
    if t is None or t == "":
        return True
    return _is_mixed_script(t)


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


# --- engines -----------------------------------------------------------------
def _build_whisper(model_name: str):
    import whisper  # imported here so the HF path works even if whisper is absent
    logger.info("Loading Whisper model '%s' ...", model_name)
    model = whisper.load_model(model_name)
    logger.info("Whisper model loaded.")
    return model


def _transcribe_whisper(model, wav_path) -> str:
    res = model.transcribe(str(wav_path), language="ur", task="transcribe",
                           fp16=False, verbose=False, beam_size=1)
    return (res.get("text") or "").strip()


def _build_hf(model_id: str):
    import os
    import torch
    # Use all available CPU cores for torch (the default is often a small subset),
    # which materially speeds up CPU inference.
    torch.set_num_threads(os.cpu_count() or 1)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    logger.info("Loading HuggingFace ASR model '%s' (CPU) ...", model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="float32"
    )
    # This 2023 fine-tune ships an outdated generation_config that is incompatible
    # with passing language=/task= to generate() (huggingface/transformers#25084).
    # The documented workaround is forced_decoder_ids, but transformers 5.x removed
    # that as a generate() kwarg (it raises "model_kwargs are not used"). The
    # modern equivalent is decoder_input_ids, built from the same prompt ids.
    forced = processor.get_decoder_prompt_ids(language="urdu", task="transcribe")
    decoder_input_ids = torch.tensor([[i for _, i in forced]], dtype=torch.long)
    logger.info("HuggingFace ASR model loaded.")
    return processor, model, decoder_input_ids


def _transcribe_hf(processor, model, decoder_input_ids, wav_path) -> str:
    import numpy as np
    import librosa
    import torch
    # defensive load + resample to 16 kHz mono
    y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            num_beams=1,        # greedy decoding — much faster on CPU, small quality tradeoff
            max_new_tokens=225,  # Whisper's typical cap for a ~20-30s segment; prevents runaway generation
        )
    # Drop the forced prefix tokens we supplied (decoder_input_ids); this 2023
    # fine-tune's tokenizer does not mark the prompt special tokens as
    # special, so skip_special_tokens leaves them in. We also strip any
    # Whisper special tokens (e.g. <|notimestamps|>) the model emits in
    # its continuation, since the old tokenizer won't skip them either.
    cont = generated[0][decoder_input_ids.shape[1]:]
    text = processor.decode(cont, skip_special_tokens=True)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return text.strip()


def _print_summary(records, skipped, reprocessed, unchanged, errors) -> None:
    line = (f"SUMMARY: total={len(records)} verified_skipped={skipped} "
            f"reprocessed={reprocessed} unchanged={unchanged} errors={errors}")
    logger.info(line)
    try:
        with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as lf:
            lf.write(f"[{_ts()}] {line}\n")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: segment transcription (background).")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        help="Whisper model name (legacy 'whisper' backend only).")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe all non-verified records (verified still protected).")
    parser.add_argument("--batch", type=int, default=20, help="Rewrite manifest every N segments.")
    args = parser.parse_args()

    setup_logging()
    backend = TRANSCRIPTION_BACKEND
    model_id = HF_ASR_MODEL_ID if backend == "hf" else args.model
    logger.info("Starting transcription (backend=%s, model=%s, force=%s)",
                backend, model_id, args.force)

    records = load_manifest()
    if not records:
        logger.error("No manifest records found at %s. Run day1_pipeline.py first.", MANIFEST_PATH)
        return

    # --- segment selection ---
    pending = []
    n_skipped_verified = 0
    n_unchanged = 0
    for r in records:
        if r.get("verified"):
            n_skipped_verified += 1
            continue
        if args.force or _needs_retranscript(r):
            pending.append(r)
        else:
            n_unchanged += 1
    total = len(pending)
    logger.info("Segments total=%d | verified(skipped)=%d | to-process=%d | unchanged=%d",
                len(records), n_skipped_verified, total, n_unchanged)
    if total == 0:
        logger.info("Nothing to transcribe.")
        _print_summary(records, n_skipped_verified, 0, n_unchanged, 0)
        return

    # --- build the selected engine ---
    if backend == "whisper":
        model = _build_whisper(args.model)
        transcribe_fn = lambda wav: _transcribe_whisper(model, wav)
    else:
        processor, model, forced_decoder_ids = _build_hf(HF_ASR_MODEL_ID)
        transcribe_fn = lambda wav: _transcribe_hf(processor, model, forced_decoder_ids, wav)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_LOG, "w", encoding="utf-8") as lf:
        lf.write(f"[{_ts()}] Transcription started. {_progress(0, total)}\n")

    done = 0
    n_reprocessed = 0
    n_errors = 0
    start = time.time()
    for r in pending:
        seg_id = r.get("segment_id")
        fname = r.get("filename")
        wav = SEGMENTS_DIR / r.get("source_type", "") / fname
        if not wav.exists():
            wav = SEGMENTS_DIR / fname
        if not wav.exists():
            log_error(logger, "transcribe", seg_id, FileNotFoundError(f"{fname} not found"))
            n_errors += 1
            done += 1
            continue
        try:
            text = transcribe_fn(wav)
            r["transcript"] = text
            r["transcript_model"] = model_id
            n_reprocessed += 1
        except Exception as e:
            log_error(logger, "transcribe", seg_id, e)
            n_errors += 1  # leave transcript as-is on failure

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
    _print_summary(records, n_skipped_verified, n_reprocessed, n_unchanged, n_errors)


def _ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
