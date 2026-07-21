@echo off
rem Day 3 — Run Whisper Large-v3 STT LoRA Fine-Tuning
rem Output is logged to processed\training.log

echo Starting Whisper Large-v3 LoRA Training...
cd /d "%~dp0"

python pipeline\train_stt_lora.py --model_name openai/whisper-large-v3 --epochs 3 --batch_size 4 > processed\training.log 2>&1

echo Training session finished or sent to background. Check processed\training.log for details.
pause
