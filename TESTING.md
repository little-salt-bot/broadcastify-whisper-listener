# Testing

## Why automated tests are limited

This project is fundamentally a **live-streaming audio transcription** tool. Its core behavior depends on:

1. **A live Broadcastify feed** — requires network access to `hls-o1.broadcastify.com` and an online feed. There is no offline/mock feed.
2. **Real-time police/fire traffic** — the feed must be actively transmitting for VAD to detect speech. Feeds are often quiet for long stretches.
3. **A Whisper model download** — the model (~1.5GB for `medium`) downloads on first run from HuggingFace.

These make deterministic unit testing of the full pipeline impractical without significant mocking infrastructure.

## What CAN be tested

The pure, network-independent pieces can and should be unit tested:

- **`get_segments()`** — parses an m3u8 playlist into segment URLs. Pure string parsing, no I/O.
- **`decode_pcm()`** — decodes a known-good MPEG-TS byte string to PCM. Deterministic given fixed input.

## How to test in the future

1. **Unit tests** for `get_segments()` and `decode_pcm()` using `pytest` with a fixture m3u8 and a small recorded `.ts` segment checked into `tests/fixtures/`.
2. **Integration test** against a live feed, gated behind an env var (e.g. `BCFY_LIVE_TEST=1`) so CI doesn't depend on network/feed availability.
3. **Golden-file test** — record a short transmission to a `.wav`, run it through the VAD+Whisper path, and assert the transcript matches an expected string.

## Current status

No automated tests are committed yet. The pipeline was verified manually against live feeds (Phoenix feed 1 and Bedford feed 41286) with the `small` and `medium` models producing correct dispatch transcripts.
