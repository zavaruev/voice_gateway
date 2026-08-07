FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libopus0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt speexdsp-ns

RUN curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

COPY main.py audio_utils.py camera_client.py engine.py .
COPY config/computer.onnx config/computer.onnx
COPY config/devices.json config/devices.json
COPY config/speaker_names.json config/speaker_names.json
COPY config/chat_id_cache.json config/chat_id_cache.json
COPY templates/ ./templates/
EXPOSE 18792 8080

CMD ["python", "-u", "main.py"]
