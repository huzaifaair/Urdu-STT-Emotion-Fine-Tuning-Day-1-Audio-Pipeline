"""
Fix code-switching: convert known English loanwords that got transliterated
into Urdu script back to their Latin spelling, across every transcript.

Only touches the ~11 confirmed CASUAL code-switch words (export, mentor,
struggle, buyer, product, project, start, point, share, level, leader).
Deliberately leaves the more "established"/assimilated loanwords
(bank, dollar, team, data, percent, professional) in Urdu script.

Before overwriting, the original transcript is preserved in a new field
`transcript_pre_codeswitch_fix`, so nothing is lost / this is reversible.

Run from pipeline/ dir:
    python fix_code_switch_apply.py             # apply the fix, write manifest
    python fix_code_switch_apply.py --dry-run   # just show counts + sample diffs
"""
from __future__ import annotations
import sys
import re
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MANIFEST_PATH, read_manifest, write_manifest, setup_logging, logger

# Urdu-script token -> English replacement.
# Includes common plural/suffix variants proactively (harmless if they never occur).
REPLACEMENTS = {
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
}

# Trailing/leading punctuation to strip before matching, then re-attach after.
PUNCT = ".,!?؟،۔()[]{}\"'"


def fix_text(text: str, counts: Counter) -> str:
    out_tokens = []
    for tok in text.split():
        lead = ""
        trail = ""
        core = tok
        while core and core[0] in PUNCT:
            lead += core[0]
            core = core[1:]
        while core and core[-1] in PUNCT:
            trail = core[-1] + trail
            core = core[:-1]

        if core in REPLACEMENTS:
            replacement = REPLACEMENTS[core]
            counts[core] += 1
            out_tokens.append(lead + replacement + trail)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--samples", type=int, default=8, help="Number of before/after examples to print.")
    args = parser.parse_args()

    setup_logging()
    records = read_manifest(MANIFEST_PATH)
    logger.info("Loaded %d records.", len(records))

    counts = Counter()
    changed = 0
    samples = []

    for rec in records:
        text = (rec.get("transcript") or "").strip()
        if not text:
            continue
        fixed = fix_text(text, counts)
        if fixed != text:
            changed += 1
            if len(samples) < args.samples:
                samples.append((rec.get("segment_id"), text, fixed))
            if not args.dry_run:
                rec["transcript_pre_codeswitch_fix"] = text
                rec["transcript"] = fixed

    logger.info("Changed %d / %d transcripts.", changed, len(records))
    print("\nReplacement counts by term:")
    for term, c in counts.most_common():
        print(f"  {term:<15} {REPLACEMENTS[term]:<15} x{c}")

    print(f"\nSample before/after ({min(len(samples), args.samples)} shown):\n")
    for seg_id, before, after in samples:
        print(f"[{seg_id}]")
        print(f"  before: {before}")
        print(f"  after:  {after}\n")

    if args.dry_run:
        logger.info("Dry run -- manifest NOT written.")
        return

    write_manifest(records, MANIFEST_PATH)
    logger.info("Manifest updated -> %s", MANIFEST_PATH)
    logger.info("Original text preserved in `transcript_pre_codeswitch_fix` field for each changed record.")


if __name__ == "__main__":
    main()



    