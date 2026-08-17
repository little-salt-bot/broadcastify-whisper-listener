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
import collections
import datetime
import glob
import io
import os
import struct
import sys
import time

import av
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
    """Decode TS segments to 16kHz mono 16-bit PCM using PyAV (in-process)."""
    data = b"".join(segments)
    if not data:
        return b""
    pcm = bytearray()
    try:
        container = av.open(io.BytesIO(data), mode="r", metadata_errors="ignore")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                pcm.extend(rf.to_ndarray().tobytes())
        container.close()
    except Exception as e:
        print(f"decode_pcm error: {e}", flush=True)
    return bytes(pcm)


def open_wav(path: str):
    """Open a new streaming WAV for appending PCM; sizes patched on close."""
    f = open(path, "wb")
    f.write(b"RIFF")
    f.write(struct.pack("<I", 36))
    f.write(b"WAVEfmt ")
    f.write(struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE,
                        SAMPLE_RATE * 2, 2, 16))
    f.write(b"data")
    f.write(struct.pack("<I", 0))
    f.flush()
    return f


def finalize_wav(f, path: str) -> None:
    """Patch the RIFF/data sizes in a streamed WAV and close it."""
    n = f.tell()
    f.seek(4)
    f.write(struct.pack("<I", n - 8))
    f.seek(40)
    f.write(struct.pack("<I", n - 44))
    f.close()


def chunk_start(now: datetime.datetime) -> datetime.datetime:
    """Round a time down to the last :00 or :30 boundary."""
    if now.minute < 30:
        return now.replace(minute=0, second=0, microsecond=0)
    return now.replace(minute=30, second=0, microsecond=0)


def new_feed_state():
    """Per-feed streaming state (buffer, VAD, seen segments)."""
    return {
        "client": httpx.Client(http2=True, headers={"User-Agent": UA}, timeout=30),
        "seen": collections.OrderedDict(),  # url -> None, ordered by insertion
        "buf": b"",
        "speech": b"",
        "silence_frames": 0,
        "in_tx": False,
        "tx_start_ts": None,
        "tx_start_chunk": None,   # recording chunk name when tx started
        "tx_start_offset": 0,     # byte offset in chunk when tx started
        "rec_f": None,       # open WAV handle for current 30-min chunk
        "rec_path": None,    # path of current chunk
        "rec_start": None,   # datetime of current chunk boundary
        "rec_last_ts": None, # wall-clock of last byte written to rec
    }


class ChunkRecorder:
    """Writes continuous per-feed audio into :00/:30 WAV chunks.

    Optionally caps total recordings size: when a chunk is finalized and
    the total exceeds max_bytes, the oldest finalized chunks are deleted
    until back under the cap. The currently-open chunk is never deleted.
    """

    def __init__(self, record_dir: str, max_bytes: int = 0):
        self.dir = record_dir
        self.max_bytes = max_bytes
        os.makedirs(self.dir, exist_ok=True)

    def _prune(self, active_path: str | None = None):
        """Delete oldest finalized chunks until total <= max_bytes."""
        if not self.max_bytes:
            return
        files = []
        for p in glob.glob(os.path.join(self.dir, "*.wav")):
            if p == active_path:
                continue  # never delete the chunk being written
            files.append((os.path.getmtime(p), p))
        files.sort()  # oldest first
        total = sum(os.path.getsize(p) for _, p in files)
        for _, p in files:
            if total <= self.max_bytes:
                break
            try:
                os.remove(p)
                total -= os.path.getsize(p)
                print(f"recordings > cap: pruned {os.path.basename(p)}", flush=True)
            except OSError:
                pass

    def _roll(self, st, feed_id: int, now: datetime.datetime):
        """Close the current chunk and open a new one if the boundary passed."""
        start = chunk_start(now)
        if st["rec_f"] is not None and st["rec_start"] == start:
            return
        if st["rec_f"] is not None:  # finalize the finished chunk
            finalize_wav(st["rec_f"], st["rec_path"])
        path = os.path.join(self.dir, f"feed_{feed_id}_{start:%Y%m%d_%H%M}.wav")
        st["rec_f"] = open_wav(path)
        st["rec_path"] = path
        st["rec_start"] = start
        st["rec_last_ts"] = None  # reset gap tracking for new chunk
        self._prune(path)

    def write(self, st, feed_id: int, pcm: bytes, now: datetime.datetime = None):
        if not pcm:
            return
        if now is None:
            now = datetime.datetime.now()
        self._roll(st, feed_id, now)
        st["rec_f"].write(pcm)
        st["rec_last_ts"] = now

    def pos(self, st):
        """Return (chunk_name, data_byte_offset) of current write position."""
        if st["rec_f"] is None or st["rec_path"] is None:
            return (None, 0)
        name = os.path.basename(st["rec_path"])
        data_bytes = st["rec_f"].tell() - 44  # subtract WAV header
        return (name, data_bytes)

    def close(self, st):
        if st["rec_f"] is not None:
            finalize_wav(st["rec_f"], st["rec_path"])
            st["rec_f"] = None
        self._prune()


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
    ap.add_argument("--record-dir", default="",
                    help="if set, record continuous audio into :00/:30 WAV chunks "
                         "here (feed_<id>_<YYYYMMDD_HHMM>.wav)")
    ap.add_argument("--record-max-gb", type=float, default=0,
                    help="cap total recordings at this many GB, deleting oldest "
                         "chunks past the cap (0 = unlimited)")
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

    def transcribe(feed_id: int, audio: bytes, stream_ts: datetime.datetime,
                   rec_chunk: str = None, rec_offset: int = 0):
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, beam_size=5, vad_filter=True)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        # Confidence: whisper exposes avg_logprob (per-token, closer to 0 =
        # higher confidence) and no_speech_prob (0-1, how likely it's noise).
        # Average logprob across segments, mapped to a 0-100 confidence.
        if segs:
            avg_lp = sum(s.avg_logprob for s in segs) / len(segs)
            ns = max(s.no_speech_prob for s in segs)
        else:
            avg_lp, ns = 0.0, 0.0
        # logprob -1.0 ~ 0.0 roughly maps to confidence 0-100; penalize
        # strongly when the model thinks there's no speech.
        conf = max(0.0, min(100.0, 100.0 * (1.0 + avg_lp) * (1.0 - ns)))
        if text:
            dur = len(audio) / (SAMPLE_RATE * 2)  # seconds of speech
            ts = stream_ts.strftime("%Y-%m-%d %H:%M:%S")
            name = feed_names.get(feed_id, f"feed {feed_id}")
            loc = f" rec = {rec_chunk}:{rec_offset}" if rec_chunk else ""
            line = f"[{ts}] {text} confidence = {conf:.0f}/100 dur = {dur:.1f}s{loc}"
            print(f"[{name}] {line}", flush=True)
            with open(os.path.join(args.log_dir, f"feed_{feed_id}.log"), "a") as f:
                f.write(line + "\n")

    feeds = {fid: new_feed_state() for fid in args.feed_ids}
    max_bytes = int(args.record_max_gb * 1024**3) if args.record_max_gb > 0 else 0
    recorder = ChunkRecorder(args.record_dir, max_bytes=max_bytes) if args.record_dir else None
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
                            st["seen"][u] = None
                        time.sleep(0.3)  # don't hammer segment fetches
                    except Exception:
                        pass

                if not seg_data:
                    continue

                pcm = decode_pcm(seg_data)
                if not pcm:
                    print(f"feed {feed_id} WARN: decoded 0 bytes", flush=True)
                    continue
                if recorder:
                    recorder.write(feeds[feed_id], feed_id, pcm)
                st["buf"] += pcm

                # VAD segmentation
                while len(st["buf"]) >= FRAME_BYTES:
                    frame = st["buf"][:FRAME_BYTES]
                    st["buf"] = st["buf"][FRAME_BYTES:]
                    if vad.is_speech(frame, SAMPLE_RATE):
                        if not st["in_tx"]:
                            st["tx_start_ts"] = datetime.datetime.now()  # stream time of first speech
                            if recorder:
                                chunk_name, byte_off = recorder.pos(st)
                                st["tx_start_chunk"] = chunk_name
                                st["tx_start_offset"] = byte_off
                        st["speech"] += frame
                        st["silence_frames"] = 0
                        st["in_tx"] = True
                    elif st["in_tx"]:
                        st["speech"] += frame
                        st["silence_frames"] += 1
                        if st["silence_frames"] * FRAME_MS >= args.silence_ms:
                            transcribe(feed_id, st["speech"], st["tx_start_ts"],
                                       st["tx_start_chunk"], st["tx_start_offset"])
                            st["speech"] = b""
                            st["in_tx"] = False
                            st["silence_frames"] = 0
                            st["tx_start_ts"] = None
                            st["tx_start_chunk"] = None
                            st["tx_start_offset"] = 0

                # keep seen bounded (evict oldest entries)
                while len(st["seen"]) > 200:
                    st["seen"].popitem(last=False)

            time.sleep(1)  # pace the round-robin across feeds

    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            for st in feeds.values():
                recorder.close(st)
        for st in feeds.values():
            st["client"].close()


if __name__ == "__main__":
    main()
