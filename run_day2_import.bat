@echo off
REM Day-2 label importer (Windows).
REM Merges processed/review/review_labels.json (exported from the review page)
REM back into processed/manifest.jsonl.
REM
REM Run from the workspace root:  run_day2_import.bat
REM   run_day2_import.bat --labels path\to\other.json   (import a different file)
cd /d "%~dp0"
python pipeline\day2.py import %*
