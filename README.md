# Broadcastify Whisper Listener

Streams a [Broadcastify](https://www.broadcastify.com) live scanner feed and transcribes each police/fire transmission with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## How it works

Broadcastify now serves audio as **HLS** and requires **HTTP/2** (HTTP/1.1 gets a 403). So the pipeline is:

```
Broadcastify HLS → httpx (HTTP/2) → ffmpeg decode → VAD splits transmissions → faster-whisper → log
```

- Fetches the `.m3u8` playlist and `.ts` segments over HTTP/2 with `httpx`
- Decodes them to 16kHz mono PCM with `ffmpeg` (local, no network)
- Uses WebRTC VAD to detect when a transmission starts and ends (squelch gaps)
- Transcribes each complete transmission with faster-whisper
- Writes each transcript to stdout and a log file

## Quick start

### Docker (recommended)

```bash
docker build -t broadcastify-whisper-listener .

# Feed 41286 = Bedford County Sheriff and Police
docker run -d \
  --name bcfy-listener \
  -v bcfy-cache:/app/.cache \
  -v bcfy-logs:/app/logs \
  broadcastify-whisper-listener 41286 --model medium --log /app/logs/scanner.log
```

### Bare metal

```bash
pip install -r requirements.txt
apt install ffmpeg
python scanner.py 41286 --model medium
```

## Usage

```
python scanner.py FEED_ID [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `FEED_ID` | (required) | Broadcastify feed ID, e.g. `41286` |
| `--model` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--log` | `scanner.log` | Output log file path |
| `--silence-ms` | `600` | Silence (ms) that ends a transmission |

## Model choice

- `small` — fast, but garbles radio jargon (ten-codes, unit IDs)
- `medium` — **recommended** for scanner audio; clean dispatch traffic, still fast on CPU
- `large-v3` — most accurate, but slow on CPU; only if you have a GPU

## Finding a feed ID

Browse [Broadcastify](https://www.broadcastify.com), open a feed, and the feed ID is in the URL: `https://www.broadcastify.com/listen/feed/41286` → `41286`.

## Output

Each transmission is logged as:

```
[2026-08-16 14:57:20] Great, one at the top of the hill. Engine 271, battalion 271, combat 272.
```

## Notes

- The `[mp3float Header missing]` line in stderr is harmless ffmpeg probe noise, not an error.
- The feed must be online; if it's quiet, no transcripts are produced (VAD stays silent).
- Whisper model downloads on first run (~1.5GB for `medium`); mount a volume for `.cache` to persist it.
