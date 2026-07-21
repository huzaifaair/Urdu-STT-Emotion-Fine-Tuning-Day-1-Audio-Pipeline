"""
Urdu Transcript Sanitizer & Post-Processor

Fixes:
1. Character encoding variants (Arabic Kaf/Yeh -> Urdu Kaaf/Yeh).
2. Common Urdu spelling errors (e.g., Talha spelled as طلہا/طلفا -> طلحہ, بلکل -> بالکل).
3. Phonetic English loanwords transcribed into Urdu characters back into Latin text
   (e.g., فکس -> fix, لیپ ٹاپ -> laptop, آنٹروپنر -> entrepreneur).

Usage:
    python pipeline/sanitize_transcripts.py             # apply changes and update manifest + splits
    python pipeline/sanitize_transcripts.py --dry-run   # preview changes without saving
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MANIFEST_PATH, read_manifest, write_manifest, setup_logging, logger

# Character normalization table
CHAR_MAP = {
    'ك': 'ک',  # Arabic Kaf -> Urdu Kaaf
    'ي': 'ی',  # Arabic Yeh -> Urdu Yeh
    'ى': 'ی',  # Alef Maksura -> Urdu Yeh
    'ٱ': 'ا',  # Alef Wasla -> Urdu Alef
}

# Single-token & phrase replacements (Urdu spelling errors + English loanwords)
TOKEN_REPLACEMENTS = {
    # Common Urdu spelling fixes
    "طلہا": "طلحہ",
    "طلفا": "طلحہ",
    "طلم": "طلحہ",
    "بلکل": "بالکل",
    "انشاءاللہ": "ان شاء اللہ",
    "انشاء اللہ": "ان شاء اللہ",
    "انشااللہ": "ان شاء اللہ",
    "الحمدللہ": "الحمد للہ",

    # Phonetic English loanwords -> Standard Latin
    "فکس": "fix",
    "فکسڈ": "fixed",
    "لیپ ٹاپ": "laptop",
    "لیپٹاپ": "laptop",
    "سستم": "system",
    "سسٹمز": "systems",
    "سسٹم": "system",
    "مارکیٹ": "market",
    "مارکیٹنگ": "marketing",
    "بزنس": "business",
    "ڈالر": "dollar",
    "ڈالرز": "dollars",
    "آن لائن": "online",
    "آنلائن": "online",
    "آڈیو": "audio",
    "ویڈیو": "video",
    "پوڈکاسٹ": "podcast",
    "پاڈکاسٹ": "podcast",
    "سپلائی چین": "supply chain",
    "آنٹروپنر": "entrepreneur",
    "انٹرپنیور": "entrepreneur",
    "انٹرپرینیور": "entrepreneur",
    "ایکسپورٹ": "export",
    "ایکسپورٹس": "exports",
    "ایکسپورٹر": "exporter",
    "منٹور": "mentor",
    "مینٹور": "mentor",
    "منٹورز": "mentors",
    "سٹرگل": "struggle",
    "پروڈکٹ": "product",
    "پروڈکٹس": "products",
    "پروجیکٹ": "project",
    "پروجیکٹس": "projects",
    "سٹارٹ": "start",
    "پوائنٹ": "point",
    "پوائنٹس": "points",
    "شیئر": "share",
    "لیول": "level",
    "لیولز": "levels",
    "لیڈر": "leader",
    "لیڈرز": "leaders",
    "بائر": "buyer",
    "بائرز": "buyers",
    "ٹیم": "team",
    "ٹیمیں": "teams",
    "ڈیٹا": "data",
    "کمپنی": "company",
    "کمپنیوں": "companies",
    "آفس": "office",
    "آفسز": "offices",
    "ایکزیکیٹیو": "executive",
    "ایگزیکٹو": "executive",
    "کال": "call",
    "کالز": "calls",
    "سروس": "service",
    "سروسز": "services",
    "کسٹمر": "customer",
    "کسٹمرز": "customers",
}

PUNCT = ".,!?؟،۔()[]{}\"'"


def normalize_chars(text: str) -> str:
    """Normalize Arabic/Urdu character variants and clean unwanted control characters."""
    for old_c, new_c in CHAR_MAP.items():
        text = text.replace(old_c, new_c)
    # Remove zero-width characters
    text = re.sub(r'[\u200c\u200d\u200e\u200f\ufeff]', '', text)
    return text


def sanitize_text(text: str, stats: Counter) -> str:
    """Apply character normalization, phrase replacements, and token mappings."""
    text = normalize_chars(text)

    # Multi-word phrase replacements first
    for phrase, replacement in [
        ("سپلائی چین", "supply chain"),
        ("آن لائن", "online"),
        ("ان شاء اللہ", "ان شاء اللہ"),
        ("انشاء اللہ", "ان شاء اللہ"),
        ("انشاءاللہ", "ان شاء اللہ"),
        ("انشااللہ", "ان شاء اللہ"),
        ("کال سینٹر", "call center"),
        ("لیپ ٹاپ", "laptop"),
    ]:
        if phrase in text:
            stats[phrase] += text.count(phrase)
            text = text.replace(phrase, replacement)

    # Word-by-word replacements
    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        lead, trail, core = "", "", tok
        while core and core[0] in PUNCT:
            lead += core[0]
            core = core[1:]
        while core and core[-1] in PUNCT:
            trail = core[-1] + trail
            core = core[:-1]

        if core in TOKEN_REPLACEMENTS:
            replacement = TOKEN_REPLACEMENTS[core]
            stats[core] += 1
            out_tokens.append(lead + replacement + trail)
        else:
            out_tokens.append(tok)

    return " ".join(out_tokens)


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Sanitize Urdu transcripts for spelling and loanwords.")
    parser.add_argument("--dry-run", action="store_true", help="Show statistics without saving manifest.")
    parser.add_argument("--samples", type=int, default=10, help="Number of before/after samples to display.")
    args = parser.parse_args()

    setup_logging()
    records = read_manifest(MANIFEST_PATH)
    if not records:
        logger.error("No manifest records found at %s", MANIFEST_PATH)
        return

    logger.info("Loaded %d manifest records.", len(records))

    stats = Counter()
    changed_count = 0
    samples = []

    for rec in records:
        text = (rec.get("transcript") or "").strip()
        if not text:
            continue
        sanitized = sanitize_text(text, stats)
        if sanitized != text:
            changed_count += 1
            if len(samples) < args.samples:
                samples.append((rec.get("segment_id"), text, sanitized))

            if not args.dry_run:
                if "transcript_pre_sanitize" not in rec:
                    rec["transcript_pre_sanitize"] = text
                rec["transcript"] = sanitized

    logger.info("Sanitized %d / %d transcripts.", changed_count, len(records))

    if stats:
        print("\nReplacement term frequency:")
        for term, cnt in stats.most_common():
            rep = TOKEN_REPLACEMENTS.get(term, term)
            print(f"  {term:<20} -> {rep:<20} x{cnt}")

    if samples:
        print(f"\nSample Before / After ({len(samples)} shown):")
        for seg_id, before, after in samples:
            print(f"\n[{seg_id}]")
            print(f"  BEFORE: {before}")
            print(f"  AFTER : {after}")

    if args.dry_run:
        logger.info("Dry-run complete. Manifest not modified.")
        return

    write_manifest(records, MANIFEST_PATH)
    logger.info("Manifest updated -> %s", MANIFEST_PATH)

    # Re-run day2 split to propagate changes into dataset splits
    try:
        from day2 import cmd_split
        split_args = argparse.Namespace()
        cmd_split(split_args)
        logger.info("Dataset splits refreshed in processed/splits/")
    except Exception as e:
        logger.warning("Could not automatically refresh splits: %s", e)


if __name__ == "__main__":
    main()
