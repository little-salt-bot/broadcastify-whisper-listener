# Broadcastify Whisper Listener
# Streams a Broadcastify scanner feed and transcribes each transmission with Whisper.

FROM python:3.12-slim

# PyAV bundles its own ffmpeg libs — no system ffmpeg needed
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner.py dashboard.py capture.py ./

# Whisper model cache and log output live here; mount volumes to persist
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache /app/logs

ENTRYPOINT ["python", "scanner.py"]
