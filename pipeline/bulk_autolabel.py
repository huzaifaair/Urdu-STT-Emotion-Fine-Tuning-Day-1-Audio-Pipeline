"""
Bulk auto-label: set accent + emotion uniformly, promote transcript_v2 (large-v3)
to be the final transcript, and mark every record verified=true.

This SKIPS manual transcript review/correction entirely. Whatever is in
transcript_v2 (falling back to the original transcript if v2 is missing)
becomes the training transcript as-is.

Run from pipeline/ dir:
    python bulk_autolabel.py
    python bulk_autolabel.py --accent karachi --emotion neutral   # override defaults
    python bulk_autolabel.py --dry-run                            # preview counts only
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    MANIFEST_PATH, ACCENTS, EMOTIONS,
    read_manifest, write_manifest, setup_logging, logger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-set accent/emotion and verify all records.")
    parser.add_argument("--accent", default="karachi", help="Accent label to apply to every record.")
    parser.add_argument("--emotion", default="neutral", help="Emotion label to apply to every record.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, write nothing.")
    args = parser.parse_args()

    setup_logging()

    if args.accent not in ACCENTS:
        logger.warning("'%s' is not in common.ACCENTS %s -- proceeding anyway, but check your schema.",
                        args.accent, ACCENTS)
    if args.emotion not in EMOTIONS:
        logger.warning("'%s' is not in common.EMOTIONS %s -- proceeding anyway, but check your schema.",
                        args.emotion, EMOTIONS)

    records = read_manifest(MANIFEST_PATH)
    logger.info("Loaded %d records.", len(records))

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    no_transcript = 0

    for rec in records:
        v2 = (rec.get("transcript_v2") or "").strip()
        base = (rec.get("transcript") or "").strip()
        final_transcript = v2 or base

        if not final_transcript:
            no_transcript += 1
            continue  # can't verify a record with no transcript at all

        rec["transcript"] = final_transcript
        rec["transcript_source"] = "auto_v2" if v2 else "auto_base"
        rec["accent"] = args.accent
        rec["emotion"] = args.emotion
        rec["review_notes"] = "bulk auto-labeled (accent/emotion uniform, transcript unverified by human)"
        rec["reviewed_at"] = now
        rec["verified"] = True
        updated += 1

    logger.info("Would update %d records (accent=%s, emotion=%s). %d skipped (no transcript at all).",
                updated, args.accent, args.emotion, no_transcript)

    if args.dry_run:
        logger.info("Dry run -- manifest NOT written.")
        return

    write_manifest(records, MANIFEST_PATH)
    logger.info("Manifest updated -> %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()