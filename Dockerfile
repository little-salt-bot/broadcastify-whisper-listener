# Broadcastify Whisper Listener
# Streams a Broadcastify scanner feed and transcribes each transmission with Whisper.

FROM python:3.12-slim

# ffmpeg needed to decode HLS .ts segments to PCM
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner.py dashboard.py .

# Whisper model cache and log output live here; mount volumes to persist
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache /app/logs

ENTRYPOINT ["python", "scanner.py"]
