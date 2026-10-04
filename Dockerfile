FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libopus0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt speexdsp-ns

RUN curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

COPY main.py audio_utils.py camera_client.py engine.py backends.py telegram_client.py vosk_wake.py .
# Whole config/ instead of individual files: the runtime caches
# (config/chat_id_cache.json) are gitignored, so a fresh clone has no such
# files and a per-file COPY fails the build. Both loaders tolerate a missing
# file (load_speaker_names/load_chat_id_cache swallow the error), and this
# also ships the default wake head computer_20260706_130638.onnx, which used
# to be available only through the compose bind mount. .dockerignore keeps
# the vosk models and the .bak_* wake backups out of the context.
COPY config/ ./config/
COPY templates/ ./templates/
EXPOSE 18792 8080

CMD ["python", "-u", "main.py"]
