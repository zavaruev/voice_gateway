FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

COPY main.py .
EXPOSE 18792 8080

CMD ["python", "-u", "main.py"]
