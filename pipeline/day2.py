"""
Day-2 labeling, verification, and dataset splitting (Stages 6-9 of the 7-day plan).

This module turns the Day-1 manifest into two labeled, split-ready datasets:

  export   Build a self-contained HTML review page (processed/review/review.html) with an
           embedded audio player + forms for each segment. A human corrects the Whisper
           draft transcript, tags the accent, labels the emotion, and marks it verified.
           Progress autosaves to the browser's localStorage; an "Export labels" button
           downloads processed/review/review_labels.json.

  import   Merge a completed review_labels.json back into manifest.jsonl. Updates each
           record with the corrected transcript (transcript_source="corrected"),
           accent, emotion, review_notes, reviewed_at, and verified=true.

  split    80/10/10 train/val/test split, stratified by (source_type, emotion), so both the
           STT dataset and the emotion dataset keep balanced class proportions. Writes:
             processed/splits/full/{train,val,test}.jsonl        (every usable record)
             processed/splits/stt/{train,val,test}.jsonl         (STT dataset)
             processed/splits/emotion/{train,val,test}.jsonl     (emotion dataset)
             processed/splits/summary.json

A record is "usable" for splitting when verified=true, its emotion is a real class
(not "unknown"), and it has a non-empty transcript.

Run it:
  python pipeline/day2.py export            # generate the review page
  python pipeline/day2.py import            # merge review_labels.json into the manifest
  python pipeline/day2.py import --labels path/to/review_labels.json
  python pipeline/day2.py split             # build the 80/10/10 splits

It follows the same conventions as day1_pipeline.py / transcribe.py (common.py paths,
UTF-8 JSONL, atomic writes, per-item error logging).
"""

from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    WORKSPACE, PROCESSED_DIR, SEGMENTS_DIR, MANIFEST_PATH,
    REVIEW_DIR, REVIEW_HTML, REVIEW_LABELS, SPLITS_DIR,
    ACCENTS, EMOTIONS, EMOTION_UNKNOWN, SPLIT_RATIOS, SPLIT_SEED,
    read_manifest, write_manifest, setup_logging, log_error, logger,
)


# ---------------------------------------------------------------------------
# export: build the HTML review page
# ---------------------------------------------------------------------------
def _audio_rel_path(rec: dict) -> str:
    """Relative (posix) path from the review HTML to the segment WAV, or '' if missing."""
    fname = rec.get("filename")
    src = rec.get("source_type") or ""
    if not fname:
        return ""
    seg = SEGMENTS_DIR / src / fname
    if not seg.exists():
        # fall back to scanning any source subfolder
        for sub in SEGMENTS_DIR.iterdir():
            cand = sub / fname
            if cand.exists():
                seg = cand
                break
    if not seg.exists():
        return ""
    try:
        rel = seg.relative_to(PROCESSED_DIR).as_posix()
    except ValueError:
        return seg.as_posix()
    return "../" + rel


def _build_review_items(records: list[dict]) -> list[dict]:
    items = []
    for r in records:
        draft = (r.get("transcript") or "").strip()
        items.append({
            "id": r.get("segment_id"),
            "audio": _audio_rel_path(r),
            "source": r.get("source_file") or "",
            "source_type": r.get("source_type") or "",
            "start": r.get("start_time"),
            "end": r.get("end_time"),
            "dur": r.get("duration_seconds"),
            "speaker": r.get("speaker_id") or "",
            "draft": draft,
        })
    return items


def cmd_export(args: argparse.Namespace) -> None:
    setup_logging()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    records = read_manifest(MANIFEST_PATH)
    if not records:
        logger.error("No manifest found at %s. Run day1_pipeline.py first.", MANIFEST_PATH)
        return

    items = _build_review_items(records)
    n_with_draft = sum(1 for it in items if it["draft"])
    n_missing_audio = sum(1 for it in items if not it["audio"])
    logger.info("Review page: %d segments (%d have a Whisper draft, %d missing audio file).",
                len(items), n_with_draft, n_missing_audio)
    if n_with_draft == 0:
        logger.warning("No transcripts present yet -- run the Whisper transcription job first "
                       "so reviewers start from a draft instead of transcribing from scratch.")

    html = _REVIEW_HTML_TEMPLATE.replace("__SEGMENTS__", json.dumps(items, ensure_ascii=False))
    html = html.replace("__ACCENTS__", json.dumps(ACCENTS, ensure_ascii=False))
    html = html.replace("__EMOTIONS__", json.dumps(EMOTIONS, ensure_ascii=False))
    html = html.replace("__GENERATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    REVIEW_HTML.write_text(html, encoding="utf-8")
    logger.info("Wrote review page -> %s", REVIEW_HTML)
    logger.info("Open it in a browser, review segments, then click 'Export labels' and save "
                "the JSON next to this file as: %s", REVIEW_LABELS)


# ---------------------------------------------------------------------------
# import: merge review_labels.json back into the manifest
# ---------------------------------------------------------------------------
def _load_labels(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("labels", [])
    return data  # tolerate a bare list


def cmd_import(args: argparse.Namespace) -> None:
    setup_logging()
    labels_path = Path(args.labels) if args.labels else REVIEW_LABELS
    if not labels_path.exists():
        logger.error("Labels file not found at %s. Export it from the review page first.", labels_path)
        return

    labels = _load_labels(labels_path)
    if not labels:
        logger.error("Labels file %s contained no entries.", labels_path)
        return

    records = read_manifest(MANIFEST_PATH)
    by_id = {r.get("segment_id"): r for r in records}

    updated = 0
    verified = 0
    missing = 0
    now = datetime.now(timezone.utc).isoformat()

    for lab in labels:
        seg_id = lab.get("segment_id")
        rec = by_id.get(seg_id)
        if rec is None:
            missing += 1
            log_error(logger, "import", str(seg_id), FileNotFoundError("segment_id not in manifest"))
            continue

        corrected = (lab.get("corrected_transcript") or "").strip()
        if corrected:
            rec["transcript"] = corrected
            rec["transcript_source"] = "corrected"
        elif rec.get("transcript_source") is None:
            rec["transcript_source"] = "draft"

        accent = lab.get("accent")
        if accent in ACCENTS:
            rec["accent"] = accent
        emotion = lab.get("emotion")
        if emotion in EMOTIONS or emotion == EMOTION_UNKNOWN:
            rec["emotion"] = emotion

        rec["review_notes"] = (lab.get("notes") or "").strip()
        rec["reviewed_at"] = now
        rec["verified"] = bool(lab.get("verified", False))

        updated += 1
        if rec["verified"]:
            verified += 1

    if updated:
        write_manifest(records, MANIFEST_PATH)
    logger.info("Imported %d labels -> manifest updated (%d verified, %d not found in manifest).",
                updated, verified, missing)
    logger.info("Manifest -> %s", MANIFEST_PATH)


# ---------------------------------------------------------------------------
# split: 80/10/10 stratified by (source_type, emotion)
# ---------------------------------------------------------------------------
def _split_counts(n: int) -> tuple[int, int, int]:
    """80/10/10 counts via largest-remainder so tiny groups still distribute fairly."""
    raw = [n * r for r in SPLIT_RATIOS]
    floor = [int(x) for x in raw]
    rem = n - sum(floor)
    # hand the leftover records to the largest fractional parts first
    order = sorted(range(3), key=lambda i: raw[i] - floor[i], reverse=True)
    for i in range(rem):
        floor[order[i % 3]] += 1
    return floor[0], floor[1], floor[2]


def _ensure_nonempty(train: list, val: list, test: list) -> tuple[list, list, list]:
    """Safety net: never leave val/test empty when there are records to spare."""
    if not val and train:
        k = max(1, len(train) // 10)
        val.extend(train[:k])
        del train[:k]
    if not test and train:
        k = max(1, len(train) // 10)
        test.extend(train[:k])
        del train[:k]
    return train, val, test


def _usable(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if not r.get("verified"):
            continue
        emo = r.get("emotion")
        if emo not in EMOTIONS:  # exclude "unknown"/null
            continue
        if not (r.get("transcript") or "").strip():
            continue
        out.append(r)
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_split(args: argparse.Namespace) -> None:
    setup_logging()
    rng = random.Random(SPLIT_SEED)

    records = read_manifest(MANIFEST_PATH)
    usable = _usable(records)
    if not usable:
        logger.error("No usable records (need verified=true, a real emotion label, and a "
                     "non-empty transcript). Run `export`+`import` first.")
        return

    # Group by (source_type, emotion) for stratification.
    groups: dict[tuple, list[dict]] = {}
    for r in usable:
        key = (r.get("source_type") or "?", r.get("emotion"))
        groups.setdefault(key, []).append(r)

    train, val, test = [], [], []
    for key, items in groups.items():
        rng.shuffle(items)
        a, b, c = _split_counts(len(items))
        train.extend(items[:a])
        val.extend(items[a:a + b])
        test.extend(items[a + b:a + b + c])

    train, val, test = _ensure_nonempty(train, val, test)

    # Write the three dataset views (full + per-task).
    _write_jsonl(SPLITS_DIR / "full" / "train.jsonl", train)
    _write_jsonl(SPLITS_DIR / "full" / "val.jsonl", val)
    _write_jsonl(SPLITS_DIR / "full" / "test.jsonl", test)
    _write_jsonl(SPLITS_DIR / "stt" / "train.jsonl", train)
    _write_jsonl(SPLITS_DIR / "stt" / "val.jsonl", val)
    _write_jsonl(SPLITS_DIR / "stt" / "test.jsonl", test)
    _write_jsonl(SPLITS_DIR / "emotion" / "train.jsonl", train)
    _write_jsonl(SPLITS_DIR / "emotion" / "val.jsonl", val)
    _write_jsonl(SPLITS_DIR / "emotion" / "test.jsonl", test)

    # Stamp the split onto the master manifest for traceability.
    split_of = {}
    for r in train:
        split_of[r["segment_id"]] = "train"
    for r in val:
        split_of[r["segment_id"]] = "val"
    for r in test:
        split_of[r["segment_id"]] = "test"
    for r in records:
        if r.get("segment_id") in split_of:
            r["split"] = split_of[r["segment_id"]]
    write_manifest(records, MANIFEST_PATH)

    # Summary (overall + per-emotion breakdown).
    def _emo_counts(subset):
        d = {e: 0 for e in EMOTIONS}
        for r in subset:
            d[r.get("emotion")] = d.get(r.get("emotion"), 0) + 1
        return d

    total = len(usable)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SPLIT_SEED,
        "ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "stratified_by": ["source_type", "emotion"],
        "total_usable": total,
        "train": len(train), "val": len(val), "test": len(test),
        "emotion_train": _emo_counts(train),
        "emotion_val": _emo_counts(val),
        "emotion_test": _emo_counts(test),
    }
    (SPLITS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Split %d usable records: train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%)",
                total, len(train), 100 * len(train) / total,
                len(val), 100 * len(val) / total,
                len(test), 100 * len(test) / total)
    logger.info("Wrote splits -> %s {full,stt,emotion}/{train,val,test}.jsonl", SPLITS_DIR)
    logger.info("Manifest updated with `split` field for each usable record.")


# ---------------------------------------------------------------------------
# HTML template for the review page
# ---------------------------------------------------------------------------
_REVIEW_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urdu STT — Day 2 Review</title>
<style>
  :root { --bg:#0f1115; --panel:#1a1d24; --line:#2a2f3a; --txt:#e6e6e6;
          --muted:#9aa3b2; --accent:#4f9dff; --ok:#3ecf8e; --warn:#f5a623; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif; background:var(--bg);
         color:var(--txt); }
  header { padding:12px 18px; border-bottom:1px solid var(--line); display:flex;
           gap:16px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  header .meta { color:var(--muted); font-size:12px; }
  #wrap { display:flex; height:calc(100vh - 52px); }
  #side { width:300px; border-right:1px solid var(--line); padding:14px; overflow:auto; }
  #side .row { display:flex; gap:8px; margin-bottom:10px; }
  #side input, #side select { width:100%; background:var(--panel); color:var(--txt);
           border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
  #prog { font-size:12px; color:var(--muted); margin:8px 0; }
  #progbar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  #progbar > div { height:100%; background:var(--ok); width:0; }
  #list { font-size:12px; }
  #list .it { padding:5px 7px; border-radius:5px; cursor:pointer; display:flex;
              justify-content:space-between; gap:8px; }
  #list .it:hover { background:var(--panel); }
  #list .it.active { background:var(--accent); color:#06101f; }
  #list .it .v { color:var(--ok); }
  #main { flex:1; padding:20px; overflow:auto; }
  .card { max-width:820px; margin:0 auto; background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:18px; }
  .card h2 { margin:0 0 4px; font-size:15px; }
  .card .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
  audio { width:100%; margin:10px 0; }
  label { display:block; font-size:12px; color:var(--muted); margin:12px 0 4px; }
  textarea, .card select { width:100%; background:#11141a; color:var(--txt);
          border:1px solid var(--line); border-radius:6px; padding:9px 10px; font-size:14px; }
  textarea { min-height:70px; resize:vertical; }
  .draft { background:#11141a; border:1px dashed var(--line); border-radius:6px; padding:9px 10px;
           color:var(--muted); white-space:pre-wrap; }
  .checks { display:flex; gap:18px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  .checks label { display:inline-flex; gap:6px; align-items:center; color:var(--txt); margin:0; }
  .nav { display:flex; gap:10px; justify-content:space-between; margin-top:18px; }
  button { background:var(--accent); color:#06101f; border:0; border-radius:6px;
           padding:8px 14px; font-weight:600; cursor:pointer; }
  button.sec { background:var(--panel); color:var(--txt); border:1px solid var(--line); }
  button:disabled { opacity:.4; cursor:default; }
  .pill { display:inline-block; padding:1px 7px; border-radius:10px; background:var(--line);
          color:var(--muted); font-size:11px; }
</style>
</head>
<body>
<header>
  <h1>Urdu STT — Day 2 Review</h1>
  <span class="meta">generated __GENERATED__</span>
  <span class="meta" id="saveState"></span>
  <span style="flex:1"></span>
  <button class="sec" id="importBtn">Import JSON</button>
  <button id="exportBtn">Export labels</button>
  <input type="file" id="importFile" accept="application/json,.json" hidden>
</header>
<div id="wrap">
  <div id="side">
    <div class="row"><input id="jump" placeholder="jump to segment id…"></div>
    <div class="row">
      <select id="filter">
        <option value="all">All segments</option>
        <option value="unverified">Only unverified</option>
        <option value="verified">Only verified</option>
        <option value="noaudio">Missing audio</option>
      </select>
    </div>
    <div id="prog"></div>
    <div id="progbar"><div></div></div>
    <div id="list"></div>
  </div>
  <div id="main"><div class="card" id="card"></div></div>
</div>

<script>
const SEGMENTS = __SEGMENTS__;
const ACCENTS = __ACCENTS__;
const EMOTIONS = __EMOTIONS__;
const STORAGE_KEY = "urdu_day2_review_v1";

// state[id] = {corrected, accent, emotion, verified, notes}
let state = {};
try { state = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch (e) { state = {}; }
let cur = 0;          // index into the filtered view
let view = [];        // current filtered list of segment objects

function initState() {
  for (const s of SEGMENTS) {
    if (!state[s.id]) {
      state[s.id] = { corrected: s.draft || "", accent: "unknown",
                      emotion: "unknown", verified: false, notes: "" };
    }
  }
}
function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  document.getElementById("saveState").textContent = "saved " + new Date().toLocaleTimeString();
}
function esc(x){ return (x==null?"":String(x)).replace(/&/g,"&amp;").replace(/</g,"&lt;")
  .replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

function applyFilter() {
  const f = document.getElementById("filter").value;
  const q = document.getElementById("jump").value.trim().toLowerCase();
  view = SEGMENTS.filter(s => {
    const st = state[s.id];
    if (f === "unverified" && st.verified) return false;
    if (f === "verified" && !st.verified) return false;
    if (f === "noaudio" && s.audio) return false;
    if (q && !s.id.toLowerCase().includes(q)) return false;
    return true;
  });
  if (cur >= view.length) cur = Math.max(0, view.length - 1);
  renderList(); renderCard();
}
function counts() {
  let v = 0; for (const s of SEGMENTS) if (state[s.id].verified) v++;
  return { total: SEGMENTS.length, verified: v };
}
function renderList() {
  const c = counts();
  const pct = c.total ? Math.round(100 * c.verified / c.total) : 0;
  document.getElementById("prog").textContent =
    c.verified + " / " + c.total + " verified  (" + pct + "%)";
  document.querySelector("#progbar > div").style.width = pct + "%";
  const list = document.getElementById("list");
  list.innerHTML = "";
  view.forEach((s, i) => {
    const st = state[s.id];
    const d = document.createElement("div");
    d.className = "it" + (i === cur ? " active" : "");
    d.innerHTML = "<span>" + esc(s.id) + "</span>" +
      (st.verified ? '<span class="v">✓</span>' : "");
    d.onclick = () => { cur = i; renderList(); renderCard(); };
    list.appendChild(d);
  });
}
function renderCard() {
  const card = document.getElementById("card");
  if (!view.length) { card.innerHTML = "<p>No segments match this filter.</p>"; return; }
  const s = view[cur];
  const st = state[s.id];
  const opts = (arr, sel) => arr.map(a =>
    '<option value="' + a + '"' + (a === sel ? " selected" : "") + ">" + a + "</option>").join("");
  card.innerHTML =
    '<h2>' + esc(s.id) + '</h2>' +
    '<div class="sub">' + esc(s.source) + ' &middot; ' + esc(s.source_type) +
      ' &middot; ' + (s.start!=null?s.start.toFixed(2):"?") + 's–' +
      (s.end!=null?s.end.toFixed(2):"?") + 's &middot; ' + esc(s.speaker) + '</div>' +
    (s.audio ? '<audio controls preload="none" src="' + esc(s.audio) + '"></audio>'
             : '<div class="sub" style="color:var(--warn)">audio file not found</div>') +
    '<label>Whisper draft (reference)</label><div class="draft">' +
      (s.draft ? esc(s.draft) : "<i>no transcript yet</i>") + '</div>' +
    '<label>Corrected transcript</label>' +
      '<textarea id="f_corrected">' + esc(st.corrected) + '</textarea>' +
    '<label>Accent</label><select id="f_accent">' + opts(ACCENTS, st.accent) + '</select>' +
    '<label>Emotion</label><select id="f_emotion">' + opts(EMOTIONS.concat(["unknown"]), st.emotion) + '</select>' +
    '<label>Reviewer notes</label><textarea id="f_notes" style="min-height:48px">' + esc(st.notes) + '</textarea>' +
    '<div class="checks">' +
      '<label><input type="checkbox" id="f_verified"' + (st.verified ? " checked" : "") + '> Verified (corrected + labeled)</label>' +
      '<span class="pill">Ctrl+S export &middot; ←/→ navigate</span>' +
    '</div>' +
    '<div class="nav">' +
      '<button class="sec" id="prev"' + (cur === 0 ? " disabled" : "") + '>← Prev</button>' +
      '<span class="sub">' + (cur + 1) + ' / ' + view.length + '</span>' +
      '<button id="next"' + (cur === view.length - 1 ? " disabled" : "") + '>Next →</button>' +
    '</div>';
  document.getElementById("prev").onclick = () => { if (cur > 0) { cur--; renderList(); renderCard(); } };
  document.getElementById("next").onclick = () => { if (cur < view.length - 1) { cur++; renderList(); renderCard(); } };
  const bind = (id, key, isCheck) => {
    const el = document.getElementById(id);
    el.oninput = () => {
      if (isCheck) st[key] = el.checked; else st[key] = el.value;
      save(); renderList();
    };
  };
  bind("f_corrected", "corrected"); bind("f_accent", "accent");
  bind("f_emotion", "emotion"); bind("f_notes", "notes"); bind("f_verified", "verified", true);
}
function exportLabels() {
  const labels = SEGMENTS.map(s => {
    const st = state[s.id];
    return { segment_id: s.id, corrected_transcript: st.corrected,
             accent: st.accent, emotion: st.emotion,
             verified: st.verified, notes: st.notes };
  });
  const blob = new Blob([JSON.stringify({ version: 1,
    exported_at: new Date().toISOString(), labels }, null, 2)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "review_labels.json";
  a.click();
  URL.revokeObjectURL(a.href);
}
function importLabels(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      const arr = Array.isArray(data) ? data : (data.labels || []);
      let n = 0;
      for (const l of arr) {
        if (state[l.segment_id]) {
          Object.assign(state[l.segment_id], {
            corrected: l.corrected_transcript != null ? l.corrected_transcript : state[l.segment_id].corrected,
            accent: l.accent != null ? l.accent : state[l.segment_id].accent,
            emotion: l.emotion != null ? l.emotion : state[l.segment_id].emotion,
            verified: !!l.verified,
            notes: l.notes != null ? l.notes : state[l.segment_id].notes,
          });
          n++;
        }
      }
      save(); applyFilter();
      alert("Imported " + n + " labels.");
    } catch (e) { alert("Could not parse JSON: " + e.message); }
  };
  reader.readAsText(file);
}

document.getElementById("exportBtn").onclick = exportLabels;
document.getElementById("importBtn").onclick = () => document.getElementById("importFile").click();
document.getElementById("importFile").onchange = (e) => {
  if (e.target.files[0]) importLabels(e.target.files[0]);
};
document.getElementById("filter").onchange = applyFilter;
document.getElementById("jump").oninput = applyFilter;
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); exportLabels(); }
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight" && cur < view.length - 1) { cur++; renderList(); renderCard(); }
  if (e.key === "ArrowLeft" && cur > 0) { cur--; renderList(); renderCard(); }
});

initState(); save(); applyFilter();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-2 Urdu STT labeling, verification & split.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Build the HTML review page for manual labeling.")
    p_exp.set_defaults(func=cmd_export)

    p_imp = sub.add_parser("import", help="Merge review_labels.json into manifest.jsonl.")
    p_imp.add_argument("--labels", default=None, help="Path to review_labels.json (default: processed/review/review_labels.json).")
    p_imp.set_defaults(func=cmd_import)

    p_split = sub.add_parser("split", help="80/10/10 split stratified by source_type + emotion.")
    p_split.set_defaults(func=cmd_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
