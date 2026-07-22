"""
Day 2 - Step 6: Train/Val/Test split (80/10/10), source-disjoint.

Why "source-disjoint" instead of strict speaker-disjoint:
  Diarization was disabled during Day 1 (USE_DIARIZATION not set), so every
  segment's speaker_id is the placeholder "SPEAKER_0" - there's no real
  speaker signal to split on. The practical equivalent that still prevents
  leakage is splitting by SOURCE FILE (podcast episode / recording) instead:
  all segments from the same source_file always land in the same split, so
  the model never sees near-identical voice/content/background-noise in both
  train and test. This is documented here so the reasoning is traceable.

What this does:
  1. Loads manifest.jsonl
  2. Drops segments flagged too_short=True (not trainable; 2 segments currently)
  3. Groups remaining segments by source_file
  4. Shuffles source_files (seeded, reproducible) and assigns each ENTIRE file
     to train/val/test using an 80/10/10 target, greedily balancing segment
     counts so splits land close to the target ratio despite files having
     different lengths
  5. Reports per-split: segment count, emotion distribution, accent
     distribution - so imbalance (e.g. the neutral-heavy podcast data) is
     visible immediately rather than discovered later
  6. Writes three manifest files: manifest_train.jsonl, manifest_val.jsonl,
     manifest_test.jsonl into processed/

Run:  python make_split.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_PATH = Path("processed/manifest.jsonl")
OUT_DIR = Path("processed")
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


def load_manifest():
    recs = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    records = load_manifest()
    total_before = len(records)

    # --- 1. drop too_short segments (not trainable) ---
    usable = [r for r in records if not r.get("too_short")]
    dropped = total_before - len(usable)

    # --- 2. group by source_file (proxy for speaker-disjoint) ---
    by_source = defaultdict(list)
    for r in usable:
        key = r.get("source_file") or r.get("filename")
        by_source[key].append(r)

    # Sort largest-first (by segment count). This is the standard "first-fit
    # decreasing" bin-packing heuristic: placing big, hard-to-place files
    # first (into whichever split has the biggest deficit) gives a much
    # closer-to-target split than a random order, which can easily starve
    # one split entirely when a few files are much larger than the rest
    # (exactly what happened with 5 large podcast episodes + 5 tiny
    # WhatsApp clips). Ties broken by a fixed seed for reproducibility.
    # Split sources into two groups and handle them differently:
    #   - "podcast" sources: few, large, homogeneous (mostly neutral) ->
    #     bin-pack by size to hit the overall 80/10/10 ratio.
    #   - "fresh_recording" sources: many small, independent recordings that
    #     hold almost ALL the non-neutral emotion labels. If these are
    #     size-bin-packed like the podcasts, they tend to all land in
    #     whichever split has the biggest deficit at that point - stranding
    #     every angry/happy/sad example in a single split (e.g. all in test,
    #     none in train), which makes the emotion classifier untrainable.
    #     Instead we ROUND-ROBIN fresh_recording sources across train/val/test
    #     so every split gets at least some emotion variety.
    # Fresh recordings hold almost ALL the non-neutral emotion labels, but
    # there are only 6 non-neutral examples in the entire dataset (3 angry,
    # 2 sad, 1 happy). Splitting 6 examples three ways is statistically
    # meaningless - whichever split doesn't get a class simply can't learn
    # or be evaluated on it either way. Given that scarcity, the more useful
    # choice is to put ALL fresh_recording sources into TRAIN, maximizing
    # the model's only chance to learn these classes at all. Val/test will
    # still be near-entirely neutral for emotion - that's a real, documented
    # limitation of this dataset (flagged since Day 1), not something this
    # split script can fix by rearranging 15 segments.
    podcast_sources = [(k, v) for k, v in by_source.items() if v[0].get("source_type") != "fresh_recording"]
    fresh_sources = [(k, v) for k, v in by_source.items() if v[0].get("source_type") == "fresh_recording"]

    total_segments = len(usable)
    rng = random.Random(SEED)

    train, val, test = [], [], []
    for source_name, segs in fresh_sources:
        train.extend(segs)

    # With only 5 podcast-episode groups (each 126-235 segments, all larger
    # than the ~90-segment val/test targets), a deficit-driven greedy bin-pack
    # tends to dump everything into train and leave val/test empty - there's
    # no way to hit exact 80/10/10 with just 5 coarse groups. The practical
    # fix: sort by size, hand the SMALLEST episode to test and the
    # next-smallest to val (this gets each closest to its target without
    # overshooting too badly), and put the remaining (larger) episodes in
    # train. The resulting ratio won't be exactly 80/10/10 - that's an
    # inherent limit of having only 5 source files, not a bug - but val and
    # test are guaranteed non-empty and every split stays source-disjoint.
    podcast_sources.sort(key=lambda kv: len(kv[1]))  # ascending: smallest first
    if len(podcast_sources) >= 2:
        test_name, test_segs = podcast_sources[0]
        val_name, val_segs = podcast_sources[1]
        test.extend(test_segs)
        val.extend(val_segs)
        remaining = podcast_sources[2:]
    else:
        remaining = podcast_sources
    for source_name, segs in remaining:
        train.extend(segs)

    # --- 4. write outputs ---
    write_jsonl(OUT_DIR / "manifest_train.jsonl", train)
    write_jsonl(OUT_DIR / "manifest_val.jsonl", val)
    write_jsonl(OUT_DIR / "manifest_test.jsonl", test)

    # --- 5. report ---
    def report(name, split):
        n = len(split)
        pct = 100 * n / total_segments if total_segments else 0
        emo = Counter(r.get("emotion") or "(missing)" for r in split)
        acc = Counter(r.get("accent") or "(missing)" for r in split)
        src = sorted(set(r.get("source_file") for r in split))
        print(f"\n--- {name} : {n} segments ({pct:.1f}%) ---")
        print(f"  source files ({len(src)}): {', '.join(src)}")
        print(f"  emotion: {dict(emo)}")
        print(f"  accent:  {dict(acc)}")

    print("=" * 70)
    print("TRAIN/VAL/TEST SPLIT SUMMARY")
    print(f"  total segments in manifest : {total_before}")
    print(f"  dropped (too_short)        : {dropped}")
    print(f"  usable segments            : {total_segments}")
    report("TRAIN", train)
    report("VAL", val)
    report("TEST", test)

    # sanity check: verify no source_file appears in more than one split
    train_srcs = set(r.get("source_file") for r in train)
    val_srcs = set(r.get("source_file") for r in val)
    test_srcs = set(r.get("source_file") for r in test)
    overlap = (train_srcs & val_srcs) | (train_srcs & test_srcs) | (val_srcs & test_srcs)
    print("\n" + "=" * 70)
    if overlap:
        print(f"WARNING: source overlap detected between splits: {overlap}")
    else:
        print("OK: no source_file appears in more than one split (source-disjoint confirmed).")
    print("=" * 70)


if __name__ == "__main__":
    main()