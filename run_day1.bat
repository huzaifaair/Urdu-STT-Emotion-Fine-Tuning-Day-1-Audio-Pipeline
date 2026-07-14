@echo off
REM Day-1 Urdu STT audio pipeline launcher (Windows).
REM Run from the workspace root:  run_day1.bat
REM
REM Usage:
REM   run_day1.bat                 full run (stages 1-4)
REM   run_day1.bat --limit-seconds 60   smoke test on first 60s of each file
REM   run_day1.bat --transcribe    also kick off background transcription when done
REM
REM Environment toggles (set before running, or export):
REM   set USE_DIARIZATION=1
REM   set HUGGINGFACE_TOKEN=hf_xxx
REM   set WHISPER_MODEL=base

cd /d "%~dp0"
python pipeline\day1_pipeline.py %*
