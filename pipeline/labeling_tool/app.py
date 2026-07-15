"""
Day-2 manual labeling tool for the Urdu STT + Emotion dataset.

Single-page Flask app (localhost) to walk the manifest segment-by-segment and:
  - play the segment audio (with 0.75x slow-down),
  - correct the Whisper draft transcript (Urdu/RTL),
  - tag accent + emotion,
  - mark verified & save (atomic write to manifest.jsonl).

Run:  python pipeline/labeling_tool/app.py
Then open the printed localhost URL.

Saves are atomic (write temp file + os.replace) and preserve every original field.
The "likely-garbage" heuristic (mixed Latin/Urdu, extreme length, repeated words) is
computed both at startup (console scan) and per-segment (red highlight in the UI).
"""

from __future__ import annotations

import os
import sys
import json
import webbrowser
from collections import Counter
from pathlib import Path

from flask import Flask, request, Response, render_template_string, jsonify

# --- paths -------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # .../Context AI
MANIFEST_PATH = WORKSPACE / "processed" / "manifest.jsonl"
SEGMENTS_DIR = WORKSPACE / "processed" / "segments"

APP = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))

# --- manifest load/save ------------------------------------------------------
_RECORDS: list[dict] = []


def load_manifest() -> list[dict]:
    recs = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def save_manifest_atomic(records: list[dict]) -> None:
    tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, MANIFEST_PATH)


# --- garbage heuristic --------------------------------------------------------
URDU_RANGES = [
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
]


def _has_script(text: str, ranges) -> bool:
    return any(any(a <= ord(ch) <= b for a, b in ranges) for ch in text)


def _has_latin(text: str) -> bool:
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)


def garbage_analysis(text: str, duration: float) -> dict:
    """Return {'flag': bool, 'reasons': [str]} for a likely-garbage draft transcript."""
    reasons: list[str] = []
    has_urdu = _has_script(text, URDU_RANGES)
    has_latin = _has_latin(text)

    if has_latin and not has_urdu:
        reasons.append("Latin-only text (no Urdu script) — likely wrong language output")
    elif has_latin and has_urdu:
        reasons.append("Mixed Latin + Urdu script")

    non_space = len(text.replace(" ", ""))
    if duration and duration > 0:
        if non_space < duration * 2:
            reasons.append(f"Very short for {duration:.0f}s audio")
        if non_space > duration * 22:
            reasons.append(f"Very long for {duration:.0f}s audio")

    tokens = text.split()
    if tokens:
        most, count = Counter(tokens).most_common(1)[0]
        if count >= 3 and count / len(tokens) > 0.4:
            reasons.append(f"Repeated word '{most}' ({count}x)")

    return {"flag": len(reasons) > 0, "reasons": reasons}


def _record_light(r: dict, index: int) -> dict:
    """Lightweight record for the frontend (full transcript + flags included)."""
    dur = float(r.get("duration_seconds", 0) or 0)
    text = r.get("transcript") or ""
    return {
        "index": index,
        "segment_id": r.get("segment_id"),
        "filename": r.get("filename"),
        "source_type": r.get("source_type"),
        "duration_seconds": dur,
        "verified": bool(r.get("verified")),
        "accent": r.get("accent"),
        "emotion": r.get("emotion"),
        "transcript": text,
        "garbage": garbage_analysis(text, dur),
    }


# --- routes ------------------------------------------------------------------
@APP.route("/")
def index():
    return render_template_string(INDEX_HTML)


@APP.route("/api/all")
def api_all():
    data = [_record_light(r, i) for i, r in enumerate(_RECORDS)]
    return jsonify({"records": data, "total": len(data),
                    "verified": sum(1 for r in data if r["verified"])})


@APP.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(force=True)
    idx = int(body["index"])
    if idx < 0 or idx >= len(_RECORDS):
        return jsonify({"ok": False, "error": "bad index"}), 400

    rec = _RECORDS[idx]
    # preserve everything; only update the editable fields
    rec["transcript"] = body.get("transcript", rec.get("transcript"))
    rec["accent"] = body.get("accent", rec.get("accent"))
    if body.get("accent_other"):
        rec["accent"] = f"Other: {body['accent_other']}"
    rec["emotion"] = body.get("emotion", rec.get("emotion"))
    if body.get("verified"):
        rec["verified"] = True

    try:
        save_manifest_atomic(_RECORDS)
    except Exception as e:  # pragma: no cover
        return jsonify({"ok": False, "error": str(e)}), 500

    light = _record_light(rec, idx)
    return jsonify({"ok": True, "record": light,
                    "verified": sum(1 for r in _RECORDS if r.get("verified")),
                    "total": len(_RECORDS)})


@APP.route("/audio")
def audio():
    fname = request.args.get("file", "")
    source = request.args.get("source", "")
    # resolve safely inside SEGMENTS_DIR (prevent path traversal)
    candidates = [
        SEGMENTS_DIR / source / fname,
        SEGMENTS_DIR / fname,
    ]
    path = None
    for c in candidates:
        try:
            c = c.resolve()
            if c.is_file() and str(c).startswith(str(SEGMENTS_DIR.resolve())):
                path = c
                break
        except Exception:
            continue
    if path is None:
        return "not found", 404
    return _send_file_range(str(path), "audio/wav")


def _send_file_range(path: str, mime: str) -> Response:
    """Minimal byte-range aware file sender (for HTML5 audio seeking)."""
    size = os.path.getsize(path)
    range_hdr = request.headers.get("Range")
    if range_hdr:
        # parse "bytes=start-end"
        try:
            rng = range_hdr.replace("bytes=", "").split(",")[0].strip()
            start_s, end_s = rng.split("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except Exception:
            start, end = 0, size - 1
        end = min(end, size - 1)
        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        resp = Response(data, 206, mimetype=mime,
                        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                  "Accept-Ranges": "bytes",
                                  "Content-Length": str(length)})
        return resp
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, 200, mimetype=mime,
                    headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})


# --- startup scan -------------------------------------------------------------
def startup_scan() -> None:
    total = len(_RECORDS)
    garbage = [r for r in _RECORDS if garbage_analysis(
        r.get("transcript") or "", float(r.get("duration_seconds", 0) or 0))["flag"]]
    verified = sum(1 for r in _RECORDS if r.get("verified"))
    print("=" * 60)
    print("DAY-2 LABELING TOOL — manifest quality scan")
    print(f"  total segments : {total}")
    print(f"  verified       : {verified}")
    print(f"  likely-garbage drafts : {len(garbage)} ({100*len(garbage)/total:.1f}%)")
    # breakdown by reason
    from collections import Counter
    rc = Counter()
    for r in garbage:
        for reason in garbage_analysis(r.get("transcript") or "",
                                      float(r.get("duration_seconds", 0) or 0))["reasons"]:
            rc[reason.split(" (")[0].split(" for")[0]] += 1
    for reason, n in rc.most_common():
        print(f"    - {reason}: {n}")
    print("=" * 60)


# --- frontend ----------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ur" dir="rtl">
<head>
<meta charset="utf-8">
<title>Urdu STT Labeling</title>
<style>
  body{font-family:Segoe UI,Tahoma,sans-serif;margin:0;background:#1e1e1e;color:#eaeaea}
  header{padding:10px 16px;background:#2d2d2d;display:flex;gap:16px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #444}
  header b{color:#ffd479}
  .wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
  .card{background:#262626;border:1px solid #3a3a3a;border-radius:8px;padding:14px}
  audio{width:100%}
  textarea{width:100%;height:160px;font-size:18px;background:#1b1b1b;color:#fff;border:1px solid #444;border-radius:6px;padding:10px;direction:rtl;text-align:right;font-family:'Noto Nastaliq Urdu',Tahoma,serif}
  textarea.garbage{background:#3a1c1c;border-color:#a33}
  .reason{color:#ff9b9b;font-size:13px;margin-top:6px}
  label{display:block;margin:10px 0 4px;color:#bbb;font-size:13px}
  select,input[type=text]{width:100%;padding:8px;background:#1b1b1b;color:#fff;border:1px solid #444;border-radius:6px;font-size:15px}
  .emo{display:flex;gap:8px;margin-top:6px}
  .emo button{flex:1;padding:10px;font-size:15px;background:#1b1b1b;color:#ccc;border:1px solid #444;border-radius:6px;cursor:pointer}
  .emo button.active{background:#2b5d8a;border-color:#3b8fd0;color:#fff}
  .emo button.sel-neutral.active{background:#3a6b3a;border-color:#5bb55b}
  .emo button.sel-happy.active{background:#7a6a1f;border-color:#d8c04a}
  .emo button.sel-angry.active{background:#7a2f2f;border-color:#d85b5b}
  .emo button.sel-sad.active{background:#2f4f7a;border-color:#5b8fd8}
  .nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .nav button{padding:9px 12px;background:#333;border:1px solid #555;color:#eee;border-radius:6px;cursor:pointer}
  .nav button.primary{background:#2b6d3a;border-color:#3fa353}
  .nav button.skip{background:#5a4a1f;border-color:#a98b3a}
  .meta{font-size:13px;color:#9aa;margin-bottom:8px;word-break:break-all}
  .bar{height:6px;background:#3a3a3a;border-radius:3px;overflow:hidden;margin-top:4px}
  .bar>i{display:block;height:100%;background:#3fa353}
  kbd{background:#444;border-radius:3px;padding:1px 5px;font-size:12px}
</style>
</head>
<body>
<header>
  <b>Urdu STT Labeling</b>
  <span id="progress">0 / 0 verified</span>
  <div class="bar" style="width:200px"><i id="bar" style="width:0%"></i></div>
  <span style="font-size:12px;color:#9aa">
    keys: <kbd>1-4</kbd> emotion · <kbd>Enter</kbd> verify+next · <kbd>s</kbd> skip · <kbd>n</kbd> next-unverified · <kbd>g</kbd> next-garbage
  </span>
</header>

<div class="wrap">
  <div class="card">
    <div class="meta" id="meta">loading…</div>
    <audio id="player" controls preload="metadata"></audio>
    <div class="nav">
      <button id="btnSlow">▶ 0.75x (slow)</button>
      <button id="btnNorm">▶ 1.0x</button>
    </div>
    <div class="nav">
      <button onclick="go(-1)">◀ Prev</button>
      <button id="btnNextUnv" onclick="nextUnverified()">Next unverified</button>
      <button id="btnNextGarbage" onclick="nextGarbage()">Next garbage</button>
      <button onclick="jumpPrompt()">Jump to ID</button>
    </div>
  </div>

  <div class="card">
    <label>Draft transcript (Urdu / RTL) — edit directly</label>
    <textarea id="transcript" dir="rtl" lang="ur"></textarea>
    <div class="reason" id="reason"></div>

    <label>Accent</label>
    <select id="accent" onchange="onAccent()">
      <option value="">—</option>
      <option>Karachi</option>
      <option>Lahori</option>
      <option>Punjabi-influenced</option>
      <option>Standard/Neutral</option>
      <option>Other</option>
    </select>
    <input type="text" id="accentOther" placeholder="If Other — specify" style="margin-top:6px;display:none">

    <label>Emotion (single click)</label>
    <div class="emo">
      <button class="sel-neutral" data-emotion="neutral" onclick="setEmotion('neutral')">1 · neutral</button>
      <button class="sel-happy" data-emotion="happy" onclick="setEmotion('happy')">2 · happy</button>
      <button class="sel-angry" data-emotion="angry" onclick="setEmotion('angry')">3 · angry</button>
      <button class="sel-sad" data-emotion="sad" onclick="setEmotion('sad')">4 · sad</button>
    </div>

    <div class="nav" style="margin-top:14px">
      <button class="primary" onclick="save(true)">✓ Mark verified &amp; next</button>
      <button class="skip" onclick="save(false)">Skip (save, no verify)</button>
    </div>
    <div class="reason" id="status" style="color:#9f9"></div>
  </div>
</div>

<script>
let RECORDS = [], IDX = 0, SLOW = false;

function refreshProgress(){
  const v = RECORDS.filter(r=>r.verified).length;
  document.getElementById('progress').textContent = `${v} / ${RECORDS.length} verified`;
  document.getElementById('bar').style.width = (100*v/RECORDS.length)+'%';
}
function render(){
  const r = RECORDS[IDX];
  document.getElementById('meta').innerHTML =
    `#${IDX} · <b>${r.segment_id}</b><br>file: ${r.filename} · ${r.duration_seconds.toFixed(1)}s · src: ${r.source_type}`;
  document.getElementById('transcript').value = r.transcript || '';
  const ta = document.getElementById('transcript');
  const g = r.garbage;
  ta.className = g.flag ? 'garbage' : '';
  document.getElementById('reason').textContent = g.flag ? '⚠ likely garbage: ' + g.reasons.join('; ') : '';
  document.getElementById('accent').value = (r.accent||'').startsWith('Other:') ? 'Other' : (r.accent||'');
  document.getElementById('accentOther').value = (r.accent||'').startsWith('Other:') ? r.accent.slice(7) : '';
  onAccent();
  document.querySelectorAll('.emo button').forEach(b=>b.classList.toggle('active', b.dataset.emotion===r.emotion));
  document.getElementById('player').src = '/audio?file='+encodeURIComponent(r.filename)+'&source='+encodeURIComponent(r.source_type);
  document.getElementById('player').playbackRate = SLOW ? 0.75 : 1.0;
  document.getElementById('status').textContent = r.verified ? '✓ already verified' : '';
}
function onAccent(){
  const a = document.getElementById('accent').value;
  document.getElementById('accentOther').style.display = a==='Other' ? 'block':'none';
}
function setEmotion(e){ RECORDS[IDX].emotion = e; render(); }
function go(d){ IDX = Math.max(0, Math.min(RECORDS.length-1, IDX+d)); render(); }
function nextUnverified(){ let i=RECORDS.findIndex((r,i)=>i>IDX && !r.verified); if(i<0) i=RECORDS.findIndex(r=>!r.verified); if(i>=0){IDX=i;render();} }
function nextGarbage(){ let i=RECORDS.findIndex((r,i)=>i>IDX && r.garbage.flag); if(i<0) i=RECORDS.findIndex(r=>r.garbage.flag); if(i>=0){IDX=i;render();} }
function jumpPrompt(){ const id=prompt('Segment ID:'); if(!id) return; const i=RECORDS.findIndex(r=>r.segment_id===id); if(i>=0){IDX=i;render();} else alert('not found'); }

function save(verify){
  const r = RECORDS[IDX];
  const body = {
    index: IDX,
    transcript: document.getElementById('transcript').value,
    accent: document.getElementById('accent').value,
    accent_other: document.getElementById('accentOther').value,
    emotion: r.emotion || '',
    verified: verify
  };
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(d=>{
      if(d.ok){
        RECORDS[IDX] = d.record;
        refreshProgress();
        document.getElementById('status').textContent = verify ? '✓ saved & verified' : 'saved (not verified)';
        if(IDX < RECORDS.length-1){ IDX++; render(); }
      } else { alert('save failed: '+d.error); }
    });
}
document.getElementById('btnSlow').onclick=()=>{SLOW=true; document.getElementById('player').playbackRate=0.75;};
document.getElementById('btnNorm').onclick=()=>{SLOW=false; document.getElementById('player').playbackRate=1.0;};
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key==='1') setEmotion('neutral');
  else if(e.key==='2') setEmotion('happy');
  else if(e.key==='3') setEmotion('angry');
  else if(e.key==='4') setEmotion('sad');
  else if(e.key==='Enter') save(true);
  else if(e.key==='s') save(false);
  else if(e.key==='n') nextUnverified();
  else if(e.key==='g') nextGarbage();
});

fetch('/api/all').then(r=>r.json()).then(d=>{
  RECORDS = d.records; refreshProgress(); if(RECORDS.length) render();
});
</script>
</body>
</html>"""


def main():
    global _RECORDS
    _RECORDS = load_manifest()
    startup_scan()
    port = int(os.environ.get("LABEL_PORT", "5000"))
    # open browser shortly after startup (best-effort)
    def _open():
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass
    from threading import Timer
    Timer(1.5, _open).start()
    APP.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
