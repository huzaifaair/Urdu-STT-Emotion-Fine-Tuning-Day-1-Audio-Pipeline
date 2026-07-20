"""
Scan transcripts for likely English-loanword transliterations (code-switching),
so you can see real examples from YOUR data before building a fix dictionary.

This doesn't guess perfectly -- it just flags segments containing short,
punctuation-adjacent Urdu tokens that often correspond to transliterated
English words (tech/business jargon tends to be short and phonetically
distinct from native Urdu vocabulary). You eyeball the output and tell me
which ones are actually English loanwords, then I'll build the exact
replace-dictionary from real examples instead of guessing spellings.

Run from pipeline/ dir:
    python find_code_switch.py
    python find_code_switch.py --limit 40      # show more/fewer examples
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MANIFEST_PATH, read_manifest, setup_logging, logger

# A short list of Urdu function/common words to IGNORE (very rough noise filter --
# not meant to be exhaustive, just cuts down obvious native-Urdu high-frequency words).
COMMON_URDU_STOPWORDS = {
    "ہے", "ہیں", "کا", "کی", "کے", "کو", "میں", "سے", "پر", "اور", "یہ", "وہ",
    "ہو", "ہوں", "نہیں", "بھی", "تو", "اس", "اپنے", "اپنی", "جو", "کہ", "ایک",
    "چاہتے", "چاہیے", "کرنا", "کریں", "کرتا", "کرتی", "تھے", "تھی", "تھا",
    "کیا", "بہت", "اگر", "رہے", "کام", "بات", "کرتے", "لیکن", "کوئی", "رہا",
    "کچھ", "ہوتا", "لیے", "چیز", "مجھے", "ہوتی", "ساتھ", "پاس", "کرنے", "کیسے",
    "چیزیں", "پھر", "سکتے", "لوگ", "اچھا", "لئے", "ایسا", "پہلے", "اندر", "طرح",
    "کسی", "وہاں", "صحیح", "ہوگا", "سکتا", "ہوتے", "ہمارے", "زیادہ",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25, help="Max example segments to print.")
    args = parser.parse_args()

    setup_logging()
    records = read_manifest(MANIFEST_PATH)
    logger.info("Loaded %d records.", len(records))

    token_counter = Counter()
    examples_by_token = {}

    for rec in records:
        text = (rec.get("transcript") or "").strip()
        if not text:
            continue
        for tok in text.split():
            tok_clean = tok.strip(".,!?؟،۔()[]{}\"'")
            if not tok_clean or tok_clean in COMMON_URDU_STOPWORDS:
                continue
            # crude heuristic: short-ish tokens (3-9 chars) are more likely to be
            # single transliterated English words than long native phrases
            if 3 <= len(tok_clean) <= 9:
                token_counter[tok_clean] += 1
                examples_by_token.setdefault(tok_clean, rec.get("segment_id"))

    logger.info("Found %d distinct short tokens (not in stopword list).", len(token_counter))
    print("\nMost frequent short tokens (review these -- flag which are English loanwords):\n")
    print(f"{'token':<20}{'count':<8}{'example segment_id'}")
    print("-" * 70)
    for tok, count in token_counter.most_common(args.limit):
        print(f"{tok:<20}{count:<8}{examples_by_token[tok]}")

    print("\nTip: paste back the tokens from this list that are actually English words")
    print("(e.g. فکس -> fix), and I'll build the exact replace-dictionary from them.")


if __name__ == "__main__":
    main()