# Urdu STT + Emotion Fine-Tuning — Day 1 Audio Pipeline

This project prepares raw Urdu audio (podcasts + fresh recordings) for fine-tuning a Whisper-based STT model and an emotion classifier, targeted at a banking call-center / IVR use case.

This README covers the full pipeline: what it does, how the folders are organized, how to set it up, how to run it, what the outputs mean, and what to do next.

---

## 1. Project Overview

**Goal:** Build a production-grade Urdu Speech-to-Text (STT) + Emotion Recognition model for a banking IVR system (Auton8/DeepPulse), eventually deployable air-gapped/on-prem.

**Day 1 scope specifically:**

1. Denoise and loudness-normalize raw audio
2. Segment audio into 15–30 second clips
3. Run speaker diarization (tag who's speaking)
4. Build a manifest (structured record) of every segment
5. Kick off draft transcription (Whisper base model) as a background job

Later days (2–7) handle manual correction, accent/emotion labeling, model training, evaluation, and deployment — see `Material_doc/` for the full 7-day plan.

---

## 2. Folder Structure

```
Context AI/
├── Fresh_rec/              ← Put new recordings here (currently empty; script handles this)
├── Material_doc/           ← Reference docs (spec review, 7-day plan, validation checklist, IVR plan)
├── Podcasts/                ← Raw source podcast audio (5 files, ~5 hours total) — DO NOT EDIT
├── pipeline/                ← All pipeline code lives here
│   ├── audio_io.py          ← Handles loading/saving audio files
│   ├── common.py             ← Shared helper functions (paths, logging, config)
│   ├── denoise.py            ← Denoising + loudness normalization logic
│   ├── diarize.py            ← Speaker diarization logic
│   ├── segment.py            ← Splits audio into 15–30s clips
│   ├── transcribe.py         ← Runs Whisper base model for draft transcripts
│   ├── day1_pipeline.py      ← Main script — runs steps 1–4 in order
│   ├── requirements.txt      ← Python dependencies
│   ├── run_day1.bat          ← One-click runner for the Day 1 pipeline
│   ├── run_transcribe.bat    ← One-click runner for the transcription job
│   └── __pycache__/          ← Auto-generated Python cache — ignore this
└── processed/                ← ALL OUTPUT LIVES HERE — never edit by hand
    ├── clean/
    │   └── podcast/          ← Denoised + normalized audio (Step 1 output)
    ├── segments/              ← 15–30s clips, ready for transcription/training (Step 2–3 output)
    ├── manifest.jsonl         ← One record per segment — the master index of the dataset
    ├── pipeline.log            ← Progress log for the Day 1 pipeline
    ├── transcription.log       ← Progress log for the transcription job
    └── errors.log              ← Any per-file errors get logged here instead of crashing
```

**Golden rule:** Only `Fresh_rec/` and `Podcasts/` should ever have files added manually. Everything inside `processed/` is generated — if something looks wrong, re-run the pipeline rather than hand-editing outputs.

---

## 3. Prerequisites

- **Python 3.9+** installed
- **ffmpeg** installed and available on PATH (required for audio processing)
- A GPU is strongly recommended for diarization and transcription — CPU will work but is much slower
- ~5–10 GB free disk space (processed audio + segments + models)

Check Python and ffmpeg are available:

```bash
python --version
ffmpeg -version
```

---

## 4. Setup (one-time)

1. Open a terminal in the `pipeline/` folder:

   ```bash
   cd pipeline
   ```

2. (Recommended) Create a virtual environment so dependencies don't clash with anything else on the machine:

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # source venv/bin/activate  # Mac/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. If diarization uses `pyannote.audio`, you may need a free Hugging Face access token — if `diarize.py` prompts for one, get it from <https://huggingface.co/settings/tokens> and follow the on-screen instructions once.

---

## 5. How to Run

### Step 1 — Run the Day 1 pipeline (denoise → segment → diarize → manifest)

From the `pipeline/` folder:

```bash
run_day1.bat
```

Or just double-click `run_day1.bat` in the file explorer.

**What this does:**

- Scans `Podcasts/` and `Fresh_rec/` for audio files (skips `Fresh_rec/` silently if empty)
- Denoises and normalizes each file → saves to `processed/clean/`
- Splits into 15–30s segments → saves to `processed/segments/`
- Runs speaker diarization and tags each segment
- Writes everything to `processed/manifest.jsonl`

**To watch progress live** while it runs, open a second terminal and run:

```bash
Get-Content ..\processed\pipeline.log -Wait
```

This step should take anywhere from a few minutes to an hour depending on total audio length and hardware.

### Step 2 — Run transcription (Whisper base draft transcripts)

Only run this **after** Step 1 has finished and `processed/segments/` is populated.

```bash
run_transcribe.bat
```

**What this does:**

- Runs Whisper base model over every segment in `processed/segments/`
- Writes the draft transcript for each segment back into `manifest.jsonl`
- Designed to run unattended — this is the "overnight" job in the 7-day plan

**To watch progress live:**

```bash
Get-Content ..\processed\transcription.log -Wait
```

You can close the terminal window and let this run in the background; check `transcription.log` later to see how far it got.

---

## 6. Understanding the Output

### `manifest.jsonl`

One JSON record per line, one line per audio segment. Example fields:

| Field | Meaning |
|---|---|
| `segment_id` | Unique ID for this clip |
| `filename` | Segment's audio filename |
| `source_file` | Which original podcast/recording it came from |
| `source_type` | `"podcast"` or `"fresh_recording"` |
| `start_time` / `end_time` | Position in the original file (seconds) |
| `duration_seconds` | Length of this segment |
| `speaker_id` | Diarization label (e.g. `SPEAKER_1`) |
| `checksum` | SHA-256 hash of the audio file (for integrity checking) |
| `dataset_version` | Currently `"v1"` |
| `crosstalk_flag` | `true` if overlapping speech was detected |
| `verified` | `false` until a human manually checks/corrects the transcript |
| `transcript` | `null` until transcription runs, then filled with Whisper's draft |

This manifest is the **single source of truth** for the dataset — every later step (manual correction, labeling, training) reads from and updates this file.

### Logs

- `pipeline.log` — step-by-step progress of denoising/segmentation/diarization
- `transcription.log` — progress of the transcription job (e.g. "150/400 segments transcribed")
- `errors.log` — any file or segment that failed gets logged here **without stopping the rest of the pipeline**. Always check this file after a run.

---

## 7. Adding New Data Later (Fresh Recordings)

Once new recordings are added to `Fresh_rec/`:

1. Just re-run `run_day1.bat` — it will pick up new files automatically and process them alongside (or instead of, depending on script logic) existing data.
2. Then re-run `run_transcribe.bat` to transcribe the new segments.

No code changes needed — this was built into the pipeline from the start.

---

## 8. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `run_day1.bat` does nothing / closes instantly | Open a terminal and run `python day1_pipeline.py` directly to see the actual error |
| Missing module errors | Re-run `pip install -r requirements.txt` inside the correct virtual environment |
| Diarization fails or hangs | Check for a Hugging Face token prompt; also confirm GPU/CPU compatibility in `diarize.py` |
| `processed/segments/` is empty after running | Check `pipeline.log` and `errors.log` — the denoise or segmentation step likely failed silently |
| Some podcasts missing from `Podcasts/` | Confirm all 5 source files are actually present before running — the pipeline only processes what it finds |
| Transcription very slow | Expected on CPU; let it run in the background overnight as designed |

---

## 9. What Comes After Day 1

Per the 7-day plan (see `Material_doc/`):

- **Day 2:** Manually correct/verify the draft transcripts in `manifest.jsonl`, tag accents, label emotions, create train/val/test splits
- **Day 3–4:** Fine-tune the STT model (LoRA/PEFT) and the emotion classifier
- **Day 5–6:** Evaluate (WER/CER, per-class F1), export models
- **Day 7:** Integration testing and sign-off

Refer to `Material_doc/` for the full spec review, validation checklist, and banking IVR plan for details on what "done" looks like at each stage.

---

## 10. Handoff Notes for New Team Members

- Don't hand-edit anything in `processed/` — if data looks wrong, fix the source (`Podcasts/` or `Fresh_rec/`) or the pipeline code, then re-run.
- `manifest.jsonl` is append/update-only in spirit — treat it as the dataset's database, not a scratch file.
- Always check `errors.log` after any run before assuming a step succeeded.
- Current dataset size and known limitations (as of Day 1): ~5 hours of podcast audio, emotion classes not yet balanced, no real banking call data yet — treat any resulting model as a first-pass baseline, not production-final.
