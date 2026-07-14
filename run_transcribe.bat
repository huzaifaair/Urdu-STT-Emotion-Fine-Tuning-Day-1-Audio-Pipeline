@echo off
REM Background Whisper draft transcription launcher (Windows).
REM Run from the workspace root:  run_transcribe.bat
REM It logs progress to processed\transcription.log (tail with: type processed\transcription.log)
REM
REM Usage:
REM   run_transcribe.bat                 resume (skip already-transcribed)
REM   run_transcribe.bat --force         re-transcribe everything
REM   run_transcribe.bat --model small   use a bigger model

cd /d "%~dp0"
python pipeline\transcribe.py %*
