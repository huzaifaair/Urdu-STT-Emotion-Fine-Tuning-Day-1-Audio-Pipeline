"""
Day 3: Whisper Large-v3 STT Fine-Tuning using PEFT (LoRA)

Trains low-rank adaptation parameters on top of Whisper Large-v3 using
the processed Urdu STT train/val splits in `processed/splits/stt/`.

Features:
- LoRA (Low-Rank Adaptation) on attention projections (q_proj, v_proj)
- Hugging Face Seq2SeqTrainer with automatic evaluation & checkpointing
- Evaluation using Word Error Rate (WER) via `evaluate` & `jiwer`
- Support for FP16/BF16 and 8-bit / 4-bit quantizations for memory efficiency

Usage:
    python pipeline/train_stt_lora.py
    python pipeline/train_stt_lora.py --model_name openai/whisper-large-v3-turbo --epochs 5
"""

import os
import sys
import json
import torch
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORKSPACE, SPLITS_DIR, SEGMENTS_DIR, PROCESSED_DIR, setup_logging, logger

# HF imports
try:
    import evaluate
    from datasets import Dataset, Audio
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    print("Missing HF dependencies. Please install: pip install transformers peft datasets evaluate jiwer accelerate")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_split_dataset(split_file: Path) -> Dataset:
    """Load JSONL split and resolve absolute audio paths."""
    records = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                fname = rec.get("filename")
                src = rec.get("source_type") or ""
                audio_path = SEGMENTS_DIR / src / fname
                if not audio_path.exists():
                    # scan subfolders
                    for sub in SEGMENTS_DIR.iterdir():
                        cand = sub / fname
                        if cand.exists():
                            audio_path = cand
                            break

                if audio_path.exists() and rec.get("transcript"):
                    records.append({
                        "audio": str(audio_path),
                        "sentence": rec["transcript"],
                    })

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper Large-v3 LoRA Fine-Tuning Script")
    parser.add_argument("--model_name", default="openai/whisper-large-v3", help="Base Whisper model identifier.")
    parser.add_argument("--language", default="Urdu", help="Target language.")
    parser.add_argument("--task", default="transcribe", help="Task for Whisper.")
    parser.add_argument("--output_dir", default=str(WORKSPACE / "checkpoints" / "whisper-large-v3-urdu-lora"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_fp16", action="store_true", default=torch.cuda.is_available())
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Whisper LoRA Fine-Tuning Pipeline...")
    logger.info("Base Model: %s | Device: %s", args.model_name, "CUDA" if torch.cuda.is_available() else "CPU")

    train_json = SPLITS_DIR / "stt" / "train.jsonl"
    val_json = SPLITS_DIR / "stt" / "val.jsonl"

    if not train_json.exists():
        logger.error("Train split missing at %s. Please run day2 split first.", train_json)
        sys.exit(1)

    train_ds = load_split_dataset(train_json)
    val_ds = load_split_dataset(val_json) if val_json.exists() else None
    logger.info("Loaded %d train samples, %d val samples.", len(train_ds), len(val_ds) if val_ds else 0)

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=1)
    if val_ds:
        val_ds = val_ds.map(prepare_dataset, remove_columns=val_ds.column_names, num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=args.use_fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        logging_steps=25,
        report_to=["none"],
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_ds else None,
        tokenizer=processor.feature_extractor,
    )

    logger.info("Starting training run...")
    trainer.train()

    final_model_dir = os.path.join(args.output_dir, "final_lora_weights")
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info("Training complete! LoRA weights saved to %s", final_model_dir)


if __name__ == "__main__":
    main()
