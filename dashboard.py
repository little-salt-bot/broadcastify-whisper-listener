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

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
LOG_DIR = "logs"
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


def read_log(path, limit=200):
    """Read the last `limit` lines of a log file."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return "".join(lines[-limit:])
    except FileNotFoundError:
        return ""


@app.route("/")
def index():
    feeds = discover_feeds()
    return render_template_string(HTML, feeds=feeds)


@app.route("/api/feeds")
def api_feeds():
    feeds = []
    for f in discover_feeds():
        feeds.append({
            "id": f["id"],
            "name": f["name"],
            "log": read_log(f["log_path"]),
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
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; padding: 24px; }
    .channel { background: #161a22; border: 1px solid #2a2f3a; border-radius: 8px; overflow: hidden; }
    .channel h2 { margin: 0; padding: 12px 16px; font-size: 15px; background: #1c212b; border-bottom: 1px solid #2a2f3a; }
    .channel pre { margin: 0; padding: 12px 16px; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; max-height: 500px; overflow-y: auto; }
    .empty { color: #6b7280; font-style: italic; }
    .updated { color: #6b7280; font-size: 12px; padding: 8px 16px; border-top: 1px solid #2a2f3a; }
  </style>
</head>
<body>
  <header><h1>Broadcastify Whisper Listener</h1></header>
  <div class="grid" id="grid"></div>
  <script>
    async function refresh() {
      const res = await fetch('/api/feeds');
      const feeds = await res.json();
      const grid = document.getElementById('grid');
      grid.innerHTML = feeds.map(f => `
        <div class="channel">
          <h2>${f.name}</h2>
          <pre>${f.log ? f.log.replace(/</g, '&lt;') : '<span class="empty">No transmissions yet</span>'}</pre>
          <div class="updated">Updated ${new Date().toLocaleTimeString()}</div>
        </div>
      `).join('');
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def main():
    global LOG_DIR, FEED_NAMES
    ap = argparse.ArgumentParser(description="Dashboard for Broadcastify Whisper Listener")
    ap.add_argument("--log-dir", default="logs", help="directory with per-feed log files")
    ap.add_argument("--feed-names", default="",
                    help="comma-separated feed_id:name pairs, e.g. 41286:Bedford,1:Phoenix")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    LOG_DIR = args.log_dir
    for pair in args.feed_names.split(","):
        if ":" in pair:
            fid, name = pair.split(":", 1)
            FEED_NAMES[fid.strip()] = name.strip()

    print(f"Dashboard on http://0.0.0.0:{args.port} (logs: {LOG_DIR})", flush=True)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
