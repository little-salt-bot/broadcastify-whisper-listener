#!/usr/bin/env python3
"""Stream a Broadcastify scanner feed and transcribe each transmission with Whisper.

Broadcastify now serves audio as HLS and requires HTTP/2 (HTTP/1.1 gets 403).
So we fetch the m3u8 + .ts segments with httpx (HTTP/2), decode them locally
with ffmpeg (stdin, no network), and run VAD + Whisper per transmission.

Usage:
    pip install faster-whisper webrtcvad-wheels numpy httpx[http2]
    apt install ffmpeg
    python3 scanner.py 41286
"""
import argparse
import datetime
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


def main():
    ap = argparse.ArgumentParser(description="Transcribe a Broadcastify scanner feed")
    ap.add_argument("feed_id", help="Broadcastify feed ID (e.g. 41286)")
    ap.add_argument("--model", default="small", help="Whisper model size (tiny/base/small/medium)")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--log", default="scanner.log", help="output log file")
    ap.add_argument("--silence-ms", type=int, default=600,
                    help="silence (ms) that ends a transmission")
    args = ap.parse_args()

    model = WhisperModel(args.model, device=args.device, compute_type="int8")
    vad = webrtcvad.Vad(2)

    client = httpx.Client(http2=True, headers={"User-Agent": UA}, timeout=30)

    def transcribe(audio: bytes, stream_ts: datetime.datetime):
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            ts = stream_ts.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {text}"
            print(line, flush=True)
            with open(args.log, "a") as f:
                f.write(line + "\n")

    print(f"Listening to feed {args.feed_id} ... (Ctrl-C to stop)", flush=True)

    seen = set()
    buf = b""
    speech = b""
    silence_frames = 0
    in_tx = False
    tx_start_ts = None  # wall-clock time the current transmission began

    try:
        while True:
            try:
                base, playlist = get_playlist(client, args.feed_id)
                seg_urls = get_segments(client, base, playlist)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    time.sleep(5)  # rate limited — back off
                else:
                    print(f"playlist error: {e}", flush=True)
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"playlist error: {e}", flush=True)
                time.sleep(2)
                continue

            new_segs = [u for u in seg_urls if u not in seen]
            if not new_segs:
                time.sleep(5)  # no new segments yet — wait for the next window
                continue

            # fetch new segments
            seg_data = []
            for u in new_segs:
                try:
                    r = client.get(u)
                    if r.status_code == 200 and r.content[:1] == b"\x47":  # TS sync byte
                        seg_data.append(r.content)
                        seen.add(u)
                    time.sleep(0.3)  # don't hammer segment fetches
                except Exception:
                    pass

            if not seg_data:
                time.sleep(5)
                continue

            pcm = decode_pcm(seg_data)
            if not pcm:
                print("WARN: decoded 0 bytes", flush=True)
                time.sleep(5)
                continue
            buf += pcm

            # VAD segmentation
            while len(buf) >= FRAME_BYTES:
                frame = buf[:FRAME_BYTES]
                buf = buf[FRAME_BYTES:]
                if vad.is_speech(frame, SAMPLE_RATE):
                    if not in_tx:
                        tx_start_ts = datetime.datetime.now()  # stream time of first speech
                    speech += frame
                    silence_frames = 0
                    in_tx = True
                elif in_tx:
                    speech += frame
                    silence_frames += 1
                    if silence_frames * FRAME_MS >= args.silence_ms:
                        transcribe(speech, tx_start_ts)
                        speech = b""
                        in_tx = False
                        silence_frames = 0
                        tx_start_ts = None

            # keep seen bounded
            if len(seen) > 200:
                seen = set(list(seen)[-100:])

    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
