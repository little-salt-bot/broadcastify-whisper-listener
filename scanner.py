#!/usr/bin/env python3
"""Stream one or more Broadcastify scanner feeds and transcribe each transmission.

Broadcastify now serves audio as HLS and requires HTTP/2 (HTTP/1.1 gets 403).
So we fetch the m3u8 + .ts segments with httpx (HTTP/2), decode them locally
with ffmpeg (stdin, no network), and run VAD + Whisper per transmission.

Multiple feeds share a single Whisper model in memory (one model, N feeds),
so adding feeds costs little RAM beyond the per-feed audio buffers.

Usage:
    pip install faster-whisper webrtcvad-wheels numpy httpx[http2]
    apt install ffmpeg
    python3 scanner.py 41286 1 32602
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

import httpx
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono = 960 bytes
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_playlist(client: httpx.Client, feed_id: int) -> tuple[str, str]:
    """Fetch the m3u8, return (base_url, playlist_text)."""
    url = f"https://hls-o1.broadcastify.com/s0/feed/{feed_id}/playlist.m3u8"
    r = client.get(url)
    r.raise_for_status()
    base = url.rsplit("/", 1)[0]
    return base, r.text


def get_segments(client: httpx.Client, base: str, playlist: str) -> list[str]:
    """Parse segment paths from the m3u8 and return absolute URLs."""
    segs = []
    for line in playlist.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            segs.append(f"{base}/{line}")
    return segs


def decode_pcm(segments: list[bytes]) -> bytes:
    """Concatenate .ts segments and decode to 16kHz mono PCM via ffmpeg stdin."""
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-f", "mpegts", "-i", "pipe:0",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    for seg in segments:
        ff.stdin.write(seg)
    ff.stdin.close()
    return ff.stdout.read()


def new_feed_state():
    """Per-feed streaming state (buffer, VAD, seen segments)."""
    return {
        "client": httpx.Client(http2=True, headers={"User-Agent": UA}, timeout=30),
        "seen": set(),
        "buf": b"",
        "speech": b"",
        "silence_frames": 0,
        "in_tx": False,
        "tx_start_ts": None,
    }


def main():
    ap = argparse.ArgumentParser(description="Transcribe Broadcastify scanner feeds")
    ap.add_argument("feed_ids", nargs="+", type=int,
                    help="Broadcastify feed IDs, e.g. 41286 1 32602")
    ap.add_argument("--model", default="small", help="Whisper model size (tiny/base/small/medium)")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--log-dir", default="logs", help="directory for per-feed log files")
    ap.add_argument("--feed-names", default="",
                    help="comma-separated feed_id:name pairs, e.g. 41286:Bedford,1:Phoenix")
    ap.add_argument("--silence-ms", type=int, default=600,
                    help="silence (ms) that ends a transmission")
    args = ap.parse_args()

    # feed_id -> human name
    feed_names = {}
    for pair in args.feed_names.split(","):
        if ":" in pair:
            fid, name = pair.split(":", 1)
            feed_names[int(fid)] = name.strip()

    os.makedirs(args.log_dir, exist_ok=True)

    model = WhisperModel(args.model, device=args.device, compute_type="int8")
    vad = webrtcvad.Vad(2)

    def transcribe(feed_id: int, audio: bytes, stream_ts: datetime.datetime):
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            ts = stream_ts.strftime("%Y-%m-%d %H:%M:%S")
            name = feed_names.get(feed_id, f"feed {feed_id}")
            line = f"[{ts}] {text}"
            print(f"[{name}] {line}", flush=True)
            with open(os.path.join(args.log_dir, f"feed_{feed_id}.log"), "a") as f:
                f.write(line + "\n")

    feeds = {fid: new_feed_state() for fid in args.feed_ids}
    print(f"Listening to feeds {args.feed_ids} ... (Ctrl-C to stop)", flush=True)

    try:
        while True:
            for feed_id, st in feeds.items():
                # fetch playlist
                try:
                    base, playlist = get_playlist(st["client"], feed_id)
                    seg_urls = get_segments(st["client"], base, playlist)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        time.sleep(5)  # rate limited — back off
                    else:
                        print(f"feed {feed_id} playlist error: {e}", flush=True)
                        time.sleep(2)
                    continue
                except Exception as e:
                    print(f"feed {feed_id} playlist error: {e}", flush=True)
                    time.sleep(2)
                    continue

                new_segs = [u for u in seg_urls if u not in st["seen"]]
                if not new_segs:
                    continue

                # fetch new segments
                seg_data = []
                for u in new_segs:
                    try:
                        r = st["client"].get(u)
                        if r.status_code == 200 and r.content[:1] == b"\x47":  # TS sync byte
                            seg_data.append(r.content)
                            st["seen"].add(u)
                        time.sleep(0.3)  # don't hammer segment fetches
                    except Exception:
                        pass

                if not seg_data:
                    continue

                pcm = decode_pcm(seg_data)
                if not pcm:
                    print(f"feed {feed_id} WARN: decoded 0 bytes", flush=True)
                    continue
                st["buf"] += pcm

                # VAD segmentation
                while len(st["buf"]) >= FRAME_BYTES:
                    frame = st["buf"][:FRAME_BYTES]
                    st["buf"] = st["buf"][FRAME_BYTES:]
                    if vad.is_speech(frame, SAMPLE_RATE):
                        if not st["in_tx"]:
                            st["tx_start_ts"] = datetime.datetime.now()  # stream time of first speech
                        st["speech"] += frame
                        st["silence_frames"] = 0
                        st["in_tx"] = True
                    elif st["in_tx"]:
                        st["speech"] += frame
                        st["silence_frames"] += 1
                        if st["silence_frames"] * FRAME_MS >= args.silence_ms:
                            transcribe(feed_id, st["speech"], st["tx_start_ts"])
                            st["speech"] = b""
                            st["in_tx"] = False
                            st["silence_frames"] = 0
                            st["tx_start_ts"] = None

                # keep seen bounded
                if len(st["seen"]) > 200:
                    st["seen"] = set(list(st["seen"])[-100:])

            time.sleep(1)  # pace the round-robin across feeds

    except KeyboardInterrupt:
        pass
    finally:
        for st in feeds.values():
            st["client"].close()


if __name__ == "__main__":
    main()
