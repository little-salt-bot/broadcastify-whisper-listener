#!/usr/bin/env python3
"""Capture one transmission from a Broadcastify feed to a WAV file.

Usage: python capture.py FEED_ID OUT.wav [--silence-ms 600]
"""
import argparse
import datetime
import subprocess
import sys
import time
import wave

import httpx
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_playlist(client, feed_id):
    url = f"https://hls-o1.broadcastify.com/s0/feed/{feed_id}/playlist.m3u8"
    r = client.get(url)
    r.raise_for_status()
    return url.rsplit("/", 1)[0], r.text


def get_segments(base, playlist):
    return [f"{base}/{l.strip()}" for l in playlist.splitlines()
            if l.strip() and not l.strip().startswith("#")]


def decode_pcm(segments):
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-f", "mpegts", "-i", "pipe:0",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for seg in segments:
        ff.stdin.write(seg)
    ff.stdin.close()
    return ff.stdout.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feed_id", type=int)
    ap.add_argument("out", help="output wav path")
    ap.add_argument("--silence-ms", type=int, default=600)
    args = ap.parse_args()

    client = httpx.Client(http2=True, headers={"User-Agent": UA}, timeout=30)
    vad = webrtcvad.Vad(2)

    seen, buf, speech = set(), b"", b""
    silence_frames, in_tx = 0, False

    print(f"Capturing feed {args.feed_id} ... waiting for a transmission (Ctrl-C to stop)", flush=True)
    try:
        while True:
            try:
                base, pl = get_playlist(client, args.feed_id)
                seg_urls = get_segments(base, pl)
            except Exception as e:
                print(f"playlist error: {e}", flush=True)
                time.sleep(2)
                continue

            new = [u for u in seg_urls if u not in seen]
            if not new:
                time.sleep(5)
                continue

            data = b""
            for u in new:
                try:
                    r = client.get(u)
                    if r.status_code == 200 and r.content[:1] == b"\x47":
                        data += r.content
                        seen.add(u)
                    time.sleep(0.3)
                except Exception:
                    pass
            if not data:
                time.sleep(5)
                continue

            pcm = decode_pcm([data])
            buf += pcm
            while len(buf) >= FRAME_BYTES:
                frame = buf[:FRAME_BYTES]
                buf = buf[FRAME_BYTES:]
                if vad.is_speech(frame, SAMPLE_RATE):
                    speech += frame
                    silence_frames = 0
                    in_tx = True
                elif in_tx:
                    speech += frame
                    silence_frames += 1
                    if silence_frames * FRAME_MS >= args.silence_ms:
                        # write wav
                        with wave.open(args.out, "wb") as w:
                            w.setnchannels(1)
                            w.setsampwidth(2)
                            w.setframerate(SAMPLE_RATE)
                            w.writeframes(speech)
                        print(f"Captured {len(speech)//2//SAMPLE_RATE}s to {args.out}", flush=True)
                        return
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
