"""Where do the 16 ms of openWakeWord.predict() actually go?

Model.predict(x) is not "run a small MLP". Per call it does:

  preprocessor(x)              -> AudioFeatures.__call__
    _buffer_raw_data(x)        -> x.tolist(), appended to a deque(maxlen=160_000)
    (if >=1280 new samples)
      _streaming_melspectrogram(n)
        _get_melspectrogram(list(self.raw_data_buffer)[-n-480:])
                              ^^^ list() over the WHOLE 10 s deque, every call,
                                  then discarded. n is only 2560.
      embedding_model.run       -> Google's speech-embedding CNN, once per 1280
  for each 1280-frame in x:  models[mdl].run(...)  -> the actual head MLP

Only the last line is the part people assume is the cost. This measures all of
them, plus what the same workload costs on openWakeWord's native 80 ms cadence,
because the gateway currently feeds 160 ms chunks.

Usage (inside the image, cwd=/app):
    python3 scripts/profile_oww_internals.py --raw /data/ambience.raw
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from engine import LocalAudioEngine  # noqa: E402

CHUNK = 2560
NATIVE = 1280   # openWakeWord's documented frame size (80 ms)
FLOOR = 600


def timeit(fn, label, per, extra=""):
    fn()                                   # warm
    t0 = time.perf_counter()
    fn()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label:46s} {ms/per:8.3f} мс/чанк  "
          f"{ms/per/1000*6.25*100:5.2f}% ядра{extra}")
    return ms / per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/ambience.raw")
    ap.add_argument("--model", default="/app/config/computer_20260706_130638.onnx")
    ap.add_argument("--seconds", type=int, default=120)
    args = ap.parse_args()

    raw = np.frombuffer(open(args.raw, "rb").read(), dtype=np.int16)
    raw = raw[:args.seconds * 16000]
    chunks = [raw[i * CHUNK:(i + 1) * CHUNK]
              for i in range(len(raw) // CHUNK)]
    chunks = [c for c in chunks
              if int(np.max(np.abs(c.astype(np.int32)))) >= FLOOR]

    eng = LocalAudioEngine(vad_threshold=0.03)
    eng.initialize_models(args.model)
    oww = eng.oww_model
    feat = oww.preprocessor
    head = list(oww.models.values())[0]
    in_name = oww.model_input_names[list(oww.models.keys())[0]]
    mel_in = feat.melspec_model.get_inputs()[0].name
    emb_in = feat.embedding_model.get_inputs()[0].name
    print(f"чанков {len(chunks)}   raw_data_buffer maxlen "
          f"{feat.raw_data_buffer.maxlen}\n")

    print("КУДА ИДЁТ ВРЕМЯ (наш путь: чанки по 2560 = 160 мс)")
    timeit(lambda: [oww.predict(c) for c in chunks],
           "Model.predict целиком", len(chunks))
    timeit(lambda: [feat._buffer_raw_data(c) for c in chunks],
           "  _buffer_raw_data (x.tolist() -> deque)", len(chunks))
    timeit(lambda: [list(feat.raw_data_buffer) for c in chunks],
           "  list(raw_data_buffer) — пересборка буфера", len(chunks),
           extra="   <-- КАЖДЫЙ вызов")
    timeit(lambda: [feat._get_melspectrogram(
        list(feat.raw_data_buffer)[-CHUNK - 480:]) for c in chunks],
           "  _get_melspectrogram (список, 3040 отсчёта)", len(chunks))
    # the embedding CNN always sees a full 76-frame mel window, taken from the
    # running mel buffer — not from a fresh window
    mel = feat.melspectrogram_buffer[-76:].astype(np.float32)[None, :, :, None]
    timeit(lambda: [feat.embedding_model.run(None, {emb_in: mel})
                    for c in chunks],
           "  embedding_model.run (CNN 76x32x1)", len(chunks))
    fb = feat.get_features(16)
    timeit(lambda: [head.run(None, {in_name: fb}) for c in chunks],
           "  head MLP run [1,16,96]", len(chunks))

    print("\nТОТ ЖЕ МАТЕРИАЛ НА РОДНОЙ ЧАСТОТЕ openWakeWord (1280 = 80 мс)")
    nat = [np.ascontiguousarray(raw[i * NATIVE:(i + 1) * NATIVE])
           for i in range(min(len(raw) // NATIVE, len(chunks) * 2))]
    timeit(lambda: [oww.predict(c) for c in nat],
           "Model.predict целиком, 80 мс", len(nat))
    timeit(lambda: [list(feat.raw_data_buffer) for c in nat],
           "  list(raw_data_buffer)", len(nat))

    print("\nПОТОЛОК openWakeWord, ЕСЛИ УБРАТЬ list(буфер) ИЗ ПУТИ")
    ring = np.asarray(feat.raw_data_buffer, dtype=np.float32)

    def mel_direct():
        for _ in chunks:
            w = ring[-CHUNK - 480:]          # numpy view, без конвертации
            feat.melspec_model.run(
                None, {mel_in: np.asarray(w, dtype=np.float32)[None, :]})
    try:
        timeit(mel_direct, "mel напрямую из numpy (без deque->list)",
               len(chunks))
    except Exception as e:
        print(f"  не измерить: {e}")

    print("\nСПРАВОЧНО: Silero VAD (он же зовётся на каждом чанке)")
    timeit(lambda: [eng._ww_vad_speech(c) for c in chunks],
           "_ww_vad_speech", len(chunks))


if __name__ == "__main__":
    main()