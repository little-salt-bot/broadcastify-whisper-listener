#!/usr/bin/env python3
"""Dashboard for Broadcastify Whisper Listener logs.

Reads per-feed log files (feed_<id>.log) and displays them in channels
labeled by feed name. Auto-refreshes.

Usage:
    pip install flask
    python dashboard.py --log-dir logs --feed-names 41286:Bedford,1:Phoenix
"""
import argparse
import glob
import os
import struct
import tempfile
import wave
from collections import deque

from flask import Flask, jsonify, render_template_string, request, send_file

app = Flask(__name__)
LOG_DIR = "logs"
RECORD_DIR = None
FEED_NAMES = {}


def discover_feeds():
    """Return list of {id, name, log_path} for every configured feed.

    Shows all feeds passed via --feed-names even if they have no log file
    yet (quiet feed = no transmissions = no log). Also picks up any
    feed_*.log not covered by --feed-names.
    """
    feeds = []
    seen = set()
    for fid, name in FEED_NAMES.items():
        path = os.path.join(LOG_DIR, f"feed_{fid}.log")
        feeds.append({"id": fid, "name": name, "log_path": path})
        seen.add(fid)
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "feed_*.log"))):
        fid = os.path.basename(path).replace("feed_", "").replace(".log", "")
        if fid not in seen:
            feeds.append({
                "id": fid,
                "name": FEED_NAMES.get(fid, f"Feed {fid}"),
                "log_path": path,
            })
    return feeds


def read_log(path, limit=200, date=None, page=0):
    """Read log lines, optionally filtered to a date and paginated backwards.

    date (YYYY-MM-DD): only keep lines on that day (prefix match).
    page: 0 = latest `limit` lines of that day, 1 = the `limit` before,
          etc. page=-1 means "no pagination, just last `limit` overall"
          (used by the live auto-refresh view).
    Returns (text, has_more) where has_more is True if older lines remain.

    Uses deque to avoid reading the entire file into memory — only keeps
    the relevant window. For date-filtered reads it still scans the file
    once (prefix match) but only retains matching lines in memory.
    """
    try:
        with open(path, "r") as f:
            if date:
                # date filter: scan once, keep only matching lines
                lines = [l for l in f if l.startswith(f"[{date}")]
                if page == -1:
                    return "".join(lines[-limit:]), False
                start = max(0, len(lines) - limit * (page + 1))
                end = len(lines) - limit * page
                has_more = start > 0
                return "".join(lines[start:end]), has_more
            else:
                # no date filter: use deque to tail the file efficiently
                if page == -1 or page == 0:
                    lines = list(deque(f, maxlen=limit))
                    return "".join(lines), False
                # for older pages without date, we need more lines
                needed = limit * (page + 1)
                lines = list(deque(f, maxlen=needed))
                start = max(0, len(lines) - limit * (page + 1))
                end = len(lines) - limit * page
                has_more = len(lines) >= needed
                return "".join(lines[start:end]), has_more
    except FileNotFoundError:
        return "", False


@app.route("/")
def index():
    feeds = discover_feeds()
    return render_template_string(HTML, feeds=feeds, record_dir=RECORD_DIR)


@app.route("/api/recordings/<date>")
def api_recordings(date):
    """List recording chunks for a date (YYYY-MM-DD)."""
    if not RECORD_DIR:
        return jsonify([])
    files = sorted(glob.glob(os.path.join(RECORD_DIR, f"*_{date}_*.wav")))
    out = []
    for p in files:
        name = os.path.basename(p)
        out.append({
            "name": name,
            "size": os.path.getsize(p),
            "url": f"/recordings/{name}",
        })
    return jsonify(out)


@app.route("/recordings/<name>")
def get_recording(name):
    """Serve a recorded WAV chunk."""
    if not RECORD_DIR or "/" in name or os.path.basename(name) != name:
        return ("", 404)
    p = os.path.join(RECORD_DIR, name)
    if not os.path.exists(p):
        return ("", 404)
    return send_file(p, mimetype="audio/wav", conditional=True)


@app.route("/clip/<name>")
def get_clip(name):
    """Extract a short segment from a WAV chunk.

    Query params (time mode): start (sec), end (sec).
    Query params (byte mode): offset (bytes), length (bytes).
    Returns a standalone WAV.
    """
    if not RECORD_DIR or "/" in name or os.path.basename(name) != name:
        return ("", 404)
    p = os.path.join(RECORD_DIR, name)
    if not os.path.exists(p):
        return ("", 404)

    with wave.open(p, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if "offset" in request.args:
            # byte mode: extract by raw byte offset and length
            try:
                byte_off = int(request.args.get("offset", 0))
                byte_len = int(request.args.get("length", 64000))
            except ValueError:
                return ("", 400)
            # align to frame boundary (sampwidth * channels bytes per frame)
            frame_size = sampwidth * channels
            start_frame = (byte_off // frame_size) * frame_size // frame_size
            wf.setpos(byte_off // frame_size)
            total_frames = wf.getnframes()
            n_frames = min(byte_len // frame_size, total_frames - (byte_off // frame_size))
            frames = wf.readframes(n_frames)
        else:
            # time mode: extract by seconds
            try:
                start = max(0, float(request.args.get("start", 0)))
                end = float(request.args.get("end", start + 4))
            except ValueError:
                return ("", 400)
            if end <= start:
                return ("", 400)
            total = wf.getnframes()
            duration = total / rate
            start_frame = int(min(start, duration) * rate)
            end_frame = int(min(end, duration) * rate)
            if end_frame <= start_frame:
                end_frame = min(start_frame + int(4 * rate), total)
                start_frame = max(0, end_frame - int(4 * rate))
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)

    # Write to temp file so send_file gets a real file with full seek support
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        with wave.open(f, "wb") as out:
            out.setframerate(rate)
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.writeframes(frames)

    from flask import after_this_request

    @after_this_request
    def cleanup(response):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return response

    return send_file(tmp, mimetype="audio/wav", download_name="clip.wav", as_attachment=False)


@app.route("/api/feeds")
def api_feeds():
    date = request.args.get("date", "") or None
    page = request.args.get("page", "0")
    try:
        page = int(page)
    except ValueError:
        page = 0
    feeds = []
    for f in discover_feeds():
        text, has_more = read_log(f["log_path"], date=date, page=page)
        feeds.append({
            "id": f["id"],
            "name": f["name"],
            "log": text,
            "has_more": has_more,
        })
    return jsonify(feeds)


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Broadcastify Whisper Listener</title>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
    header { padding: 16px 24px; background: #161a22; border-bottom: 1px solid #2a2f3a; }
    h1 { margin: 0; font-size: 20px; }
    .controls { padding: 16px 24px; background: #161a22; border-bottom: 1px solid #2a2f3a; display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }
    .controls label { font-size: 14px; color: #9aa0a6; display: flex; align-items: center; gap: 8px; }
    .controls input { background: #0f1115; color: #e6e6e6; border: 1px solid #2a2f3a; padding: 6px 10px; border-radius: 6px; font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; padding: 24px; }
    .channel { background: #161a22; border: 1px solid #2a2f3a; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }
    .channel h2 { margin: 0; padding: 12px 16px; font-size: 15px; background: #1c212b; border-bottom: 1px solid #2a2f3a; }
    .channel pre { margin: 0; padding: 12px 16px; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; height: 420px; overflow-y: auto; box-sizing: border-box; }
    .tline { cursor: pointer; padding: 2px 4px; border-radius: 4px; display: block; }
    .tline:hover { background: #1c212b; }
    .tline.selected { background: #233055; }
    .empty { color: #6b7280; font-style: italic; }
    .foot { color: #6b7280; font-size: 12px; padding: 8px 16px; border-top: 1px solid #2a2f3a; display: flex; justify-content: space-between; align-items: center; }
    .foot button { background: #1c212b; color: #e6e6e6; border: 1px solid #2a2f3a; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }
    .foot button:disabled { opacity: 0.4; cursor: default; }
    .rec-list { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
    .rec-list li { background: #1c212b; border: 1px solid #2a2f3a; border-radius: 6px; padding: 8px 12px; font-size: 13px; }
    .rec-list audio { display: block; margin-top: 6px; height: 32px; }
    .rec-meta { color: #6b7280; font-size: 11px; margin-top: 4px; }
    .hidden { display: none; }
    #player { position: fixed; bottom: 0; left: 0; right: 0; background: #161a22; border-top: 1px solid #2a2f3a; padding: 12px 24px; display: none; align-items: center; gap: 16px; }
    #player audio { flex: 1; height: 36px; }
    #player .plabel { font-size: 13px; color: #9aa0a6; min-width: 200px; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #player .pclose { cursor: pointer; color: #6b7280; font-size: 18px; }
    #player .pbtn { cursor: pointer; background: #233055; border: none; color: #e6e6e6; border-radius: 6px; padding: 8px 12px; font-size: 14px; white-space: nowrap; }
    #player .pbtn:disabled { opacity: 0.4; cursor: default; }
    #player .ptime { font-size: 12px; color: #9aa0a6; min-width: 70px; text-align: center; }
    #player .pbar { flex: 1; height: 6px; background: #2a2f3a; border-radius: 3px; cursor: pointer; position: relative; }
    #player .pbar-fill { height: 100%; background: #70A0AF; border-radius: 3px; width: 0%; transition: width 0.1s linear; }
  </style>
</head>
<body>
  <header><h1>Broadcastify Whisper Listener</h1></header>
  <div class="controls">
    <label>Date: <input type="date" id="datePicker"></label>
    <label>Page: <button id="prevBtn">&#9664; Older</button> <span id="pageLabel">0</span></label>
  </div>
  <div class="controls" id="recPanel">
    <h2 style="margin:0;font-size:15px;">Recordings</h2>
    <ul class="rec-list" id="recList"></ul>
  </div>
  <div class="grid" id="grid"></div>
  <div id="player">
    <span class="plabel" id="plabel"></span>
    <button class="pbtn" id="pbtn" disabled>&#9654; Play</button>
    <div class="pbar" id="pbar"><div class="pbar-fill" id="pbarFill"></div></div>
    <span class="ptime" id="ptime">0:00 / 0:00</span>
    <span class="pclose" onclick="closePlayer()">&#10006;</span>
  </div>
  <script>
    const datePicker = document.getElementById('datePicker');
    const prevBtn = document.getElementById('prevBtn');
    const pageLabel = document.getElementById('pageLabel');
    const player = document.getElementById('player');
    const plabel = document.getElementById('plabel');
    const pbtn = document.getElementById('pbtn');
    const pbar = document.getElementById('pbar');
    const pbarFill = document.getElementById('pbarFill');
    const ptime = document.getElementById('ptime');
    let page = 0;
    let audioCtx = null;
    let currentSrc = null;
    let currentBuffer = null;
    let playingSrc = null;
    let playStartOffset = 0;
    let playStartTime = 0;
    let clipDuration = 0;
    let rafId = null;

    datePicker.value = new Date().toISOString().slice(0, 10);
    datePicker.addEventListener('change', () => { page = 0; refresh(); loadRecordings(); });
    prevBtn.addEventListener('click', () => { page += 1; refresh(); });

    function fmtTime(s) {
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return m + ':' + String(sec).padStart(2, '0');
    }

    function tsToChunk(feedId, ts) {
      const d = new Date(ts.replace(' ', 'T'));
      const mm = d.getMinutes();
      const ss = d.getSeconds();
      const chunkMM = mm < 30 ? '00' : '30';
      const chunk = `feed_${feedId}_${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}_${String(d.getHours()).padStart(2,'0')}${chunkMM}`;
      const offset = (mm < 30 ? mm : mm - 30) * 60 + ss;
      return { chunk, offset };
    }

    function stopPlayback() {
      if (playingSrc) { try { playingSrc.stop(); } catch(e){} playingSrc = null; }
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      pbtn.innerHTML = '&#9654; Play';
    }

    function updateProgress() {
      if (!playingSrc) return;
      const elapsed = playStartOffset + (audioCtx.currentTime - playStartTime);
      if (elapsed >= clipDuration) {
        stopPlayback();
        pbarFill.style.width = '100%';
        ptime.textContent = fmtTime(clipDuration) + ' / ' + fmtTime(clipDuration);
        return;
      }
      pbarFill.style.width = (elapsed / clipDuration * 100) + '%';
      ptime.textContent = fmtTime(elapsed) + ' / ' + fmtTime(clipDuration);
      rafId = requestAnimationFrame(updateProgress);
    }

    function startPlayback(offset) {
      stopPlayback();
      if (!currentBuffer || !audioCtx) return;
      playingSrc = audioCtx.createBufferSource();
      playingSrc.buffer = currentBuffer;
      playingSrc.connect(audioCtx.destination);
      playStartOffset = offset || 0;
      playStartTime = audioCtx.currentTime;
      playingSrc.start(0, playStartOffset);
      playingSrc.onended = () => { if (playingSrc) { stopPlayback(); } };
      pbtn.innerHTML = '&#9646;&#9646; Pause';
      updateProgress();
    }

    pbtn.addEventListener('click', () => {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') { audioCtx.resume(); }
      if (playingSrc) {
        stopPlayback();
        pbarFill.style.width = (playStartOffset / clipDuration * 100) + '%';
        ptime.textContent = fmtTime(playStartOffset) + ' / ' + fmtTime(clipDuration);
      } else {
        startPlayback(playStartOffset >= clipDuration ? 0 : playStartOffset);
      }
    });

    pbar.addEventListener('click', (e) => {
      if (!currentBuffer) return;
      const rect = pbar.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      const seekTo = pct * clipDuration;
      if (playingSrc) { startPlayback(seekTo); } 
      else { playStartOffset = seekTo; pbarFill.style.width = (pct*100)+'%'; ptime.textContent = fmtTime(seekTo) + ' / ' + fmtTime(clipDuration); }
    });

    async function playClip(feedId, ts, dur, recChunk, recOffset, text) {
      let url;
      if (recChunk) {
        // byte mode: exact audio from the recording
        const byteLen = Math.ceil(dur * 16000 * 2); // 16kHz mono 16-bit
        url = `/clip/${recChunk}?offset=${recOffset}&length=${byteLen}&_=${Date.now()}`;
      } else {
        // fallback: time mode (old log lines without rec=)
        const { chunk, offset } = tsToChunk(feedId, ts);
        const start = Math.max(0, offset - 0.5);
        const end = offset + dur + 0.5;
        url = `/clip/${chunk}.wav?start=${start}&end=${end}&_=${Date.now()}`;
      }
      plabel.textContent = text.slice(0, 80);
      player.style.display = 'flex';
      stopPlayback();
      pbtn.disabled = true;
      pbtn.innerHTML = '...';
      ptime.textContent = '0:00 / 0:00';
      pbarFill.style.width = '0%';
      currentSrc = url;
      currentBuffer = null;

      try {
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        const res = await fetch(url);
        const buf = await res.arrayBuffer();
        currentBuffer = await audioCtx.decodeAudioData(buf);
        clipDuration = currentBuffer.duration;
        playStartOffset = 0;
        pbtn.disabled = false;
        pbtn.innerHTML = '&#9654; Play';
        ptime.textContent = '0:00 / ' + fmtTime(clipDuration);
      } catch(e) {
        pbtn.innerHTML = 'Error';
        ptime.textContent = 'Failed to load';
        console.error('clip error', e);
      }
    }

    window.closePlayer = function() {
      stopPlayback();
      player.style.display = 'none';
      document.querySelectorAll('.tline.selected').forEach(el => el.classList.remove('selected'));
    };

    async function refresh() {
      const date = datePicker.value;
      const res = await fetch('/api/feeds?date=' + date + '&page=' + page);
      const feeds = await res.json();
      const grid = document.getElementById('grid');
      // remember scroll positions
      const scrollMap = {};
      grid.querySelectorAll('pre').forEach(pre => {
        const name = pre.previousElementSibling ? pre.previousElementSibling.textContent : '';
        scrollMap[name] = { top: pre.scrollTop, atBottom: pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 5 };
      });
      grid.innerHTML = feeds.map(f => {
        if (!f.log) {
          return `<div class="channel">
            <h2>${f.name}</h2>
            <pre><span class="empty">No transmissions this day</span></pre>
            <div class="foot"><span>Updated ${new Date().toLocaleTimeString()}</span>${f.has_more ? '<button onclick="loadOlder()">Older &#9664;</button>' : ''}</div>
          </div>`;
        }
        // parse each line: [YYYY-MM-DD HH:MM:SS] text confidence = N/100 dur = N.Ns rec = chunk:offset
        const lines = f.log.split('\\n').filter(l => l.trim());
        const html = lines.map(l => {
          const m = l.match(/^\\[(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})\\]\\s*(.*)$/);
          if (!m) return `<span class="tline">${l.replace(/</g, '&lt;')}</span>`;
          const ts = m[1];
          const rest = m[2];
          const dm = rest.match(/dur\\s*=\\s*([\\d.]+)s/);
          const dur = dm ? parseFloat(dm[1]) : 4;
          const rm = rest.match(/rec\\s*=\\s*([^:]+):(\\d+)/);
          const recChunk = rm ? rm[1] : '';
          const recOffset = rm ? parseInt(rm[2]) : 0;
          return `<span class="tline" data-feed="${f.id}" data-ts="${ts}" data-dur="${dur}" data-rec="${recChunk}" data-offset="${recOffset}">${l.replace(/</g, '&lt;')}</span>`;
        }).join('');
        return `<div class="channel">
          <h2>${f.name}</h2>
          <pre>${html}</pre>
          <div class="foot"><span>Updated ${new Date().toLocaleTimeString()}</span>${f.has_more ? '<button onclick="loadOlder()">Older &#9664;</button>' : ''}</div>
        </div>`;
      }).join('');
      pageLabel.textContent = page;
      prevBtn.disabled = (page === 0);
      // scroll each pre to bottom (newest at bottom) or restore position
      grid.querySelectorAll('pre').forEach(pre => {
        const name = pre.previousElementSibling ? pre.previousElementSibling.textContent : '';
        const prev = scrollMap[name];
        if (prev && !prev.atBottom) {
          pre.scrollTop = prev.top;
        } else {
          pre.scrollTop = pre.scrollHeight;
        }
      });
    }

    window.loadOlder = function() { page += 1; refresh(); };
    // delegated click handler for transcript lines
    document.addEventListener('click', (e) => {
      const el = e.target.closest('.tline');
      if (!el || !el.dataset.feed) return;
      document.querySelectorAll('.tline.selected').forEach(s => s.classList.remove('selected'));
      el.classList.add('selected');
      const feedId = el.dataset.feed;
      const ts = el.dataset.ts;
      const dur = parseFloat(el.dataset.dur) || 4;
      const recChunk = el.dataset.rec || '';
      const recOffset = parseInt(el.dataset.offset) || 0;
      const text = el.textContent.replace(/^\[.*?\]\s*/, '').replace(/dur\s*=\s*[\d.]+s/, '').replace(/rec\s*=\s*[^<]+/, '').slice(0, 80);
      playClip(feedId, ts, dur, recChunk, recOffset, text);
    });

    async function loadRecordings() {
      const date = datePicker.value;
      const res = await fetch('/api/recordings/' + date);
      const recs = await res.json();
      const recList = document.getElementById('recList');
      if (!recList) return;
      recList.innerHTML = recs.length ? recs.map(r => `
        <li>
          <div>${r.name} <span class="rec-meta">(${(r.size/1024).toFixed(0)} KB)</span></div>
          <audio controls preload="none" src="${r.url}"></audio>
        </li>
      `).join('') : '<li class="empty">No recordings for this date</li>';
    }

    refresh();
    loadRecordings();
    setInterval(() => { if (page === 0) refresh(); }, 5000);
  </script>
</body>
</html>
"""


def main():
    global LOG_DIR, FEED_NAMES, RECORD_DIR
    ap = argparse.ArgumentParser(description="Dashboard for Broadcastify Whisper Listener")
    ap.add_argument("--log-dir", default="logs", help="directory with per-feed log files")
    ap.add_argument("--feed-names", default="",
                    help="comma-separated feed_id:name pairs, e.g. 41286:Bedford,1:Phoenix")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--record-dir", default="",
                    help="directory with recorded WAV chunks (feed_<id>_<YYYYMMDD_HHMM>.wav)")
    args = ap.parse_args()

    LOG_DIR = args.log_dir
    if args.record_dir:
        RECORD_DIR = args.record_dir
    for pair in args.feed_names.split(","):
        if ":" in pair:
            fid, name = pair.split(":", 1)
            FEED_NAMES[fid.strip()] = name.strip()

    print(f"Dashboard on http://0.0.0.0:{args.port} (logs: {LOG_DIR})", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
