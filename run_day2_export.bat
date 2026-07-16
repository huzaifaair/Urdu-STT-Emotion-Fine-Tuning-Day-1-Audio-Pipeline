@echo off
REM Day-2 review page generator (Windows).
REM Builds processed/review/review.html for manual transcript correction +
REM accent/emotion labeling. Open it in a browser, then "Export labels".
REM
REM Run from the workspace root:  run_day2_export.bat
cd /d "%~dp0"
python pipeline\day2.py export
