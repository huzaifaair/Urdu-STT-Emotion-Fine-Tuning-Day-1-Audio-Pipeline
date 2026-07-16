@echo off
REM Day-2 dataset splitter (Windows).
REM Builds an 80/10/10 train/val/test split (stratified by source_type + emotion)
REM under processed/splits/ for both the STT and emotion datasets.
REM
REM Run from the workspace root:  run_day2_split.bat
REM (Requires verified + emotion-labeled records from export/import first.)
cd /d "%~dp0"
python pipeline\day2.py split
