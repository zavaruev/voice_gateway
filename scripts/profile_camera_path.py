"""Per-stage CPU cost of one camera's audio path, so 8 rooms can be budgeted.

Numbers come from the live container, not from theory: /data/ambience.raw is a
real 16 kHz mono mic capture from the livingroom with the TV playing.

Every stage is timed over the SAME chunks the gateway sees, because the total
is only meaningful as a sum of the parts that actually run per 160 ms chunk:

  rms/peak stats  -> _is_echo 44-offset correlation ring
  Silero VAD      -> openWakeWord predict (fbank + MLP, split out)
  AGC rescale

Usage (inside the image, cwd=/app):
    python3 scripts/profile_camera_path.py --raw /data/ambience.raw
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from engine import LocalAudioEngine  # noqa: E402

CHUNK = 2560          # 160 ms — what _drain_vad_buf carves out
AGC_TARGET = 4000     # livingroom per-room wake target
PEAK_FLOOR = 600      # below this the model is not called at all
CHUNKS_PER_SEC = 1000 * 16000 / CHUNK / 1000  # 6.25/s per camera


def bench(fn, n_warm=3):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/ambience.raw")
    ap.add_argument("--model", default="/app/config/computer_20260706_130638.onnx")
    ap.add_argument("--seconds", type=int, default=120)
    args = ap.parse_args()

    raw = np.frombuffer(open(args.raw, "rb").read(), dtype=np.int16)
    raw = raw[:args.seconds * 16000]
    n = len(raw) // CHUNK
    chunks = [raw[i * CHUNK:(i + 1) * CHUNK] for i in range(n)]
    # chunks that actually reach the model (peak floor)
    live = [c for c in chunks if int(np.max(np.abs(c.astype(np.int32)))) >= PEAK_FLOOR]
    print(f"аудио {len(raw)/16000:.0f}s, чанков {n}, "
          f"дошедших до модели {len(live)} ({100*len(live)/max(n,1):.0f}%)")

    eng = LocalAudioEngine(vad_threshold=0.03)
    eng.initialize_models(args.model)

    print(f"\n{'этап':44s} {'мс/чанк':>9} {'% ядра на 1 камеру':>22} "
          f"{'на 8 камер':>12}")

    def row(name, ms, per_chunk_all=True):
        mpc = ms / max(n, 1) if per_chunk_all else ms / max(len(live), 1)
        print(f"{name:44s} {mpc*1:9.4f} {mpc/10*CHUNKS_PER_SEC*100:21.2f}% "
              f"{mpc/10*CHUNKS_PER_SEC*100*8:11.2f}%")

    # --- 1. cheap stats
    def stats():
        for c in chunks:
            s = c.astype(np.int32)
            _ = int(np.max(np.abs(s))), \
                float(np.sqrt((s.astype(np.int64) ** 2).mean()))
    row("numpy peak+rms", bench(stats))

    # --- 2. AGC rescale
    def agc():
        for c in live:
            pk = int(np.max(np.abs(c.astype(np.int32))))
            np.clip(c.astype(np.float32) * (AGC_TARGET / pk), -32768, 32767).astype(np.int16)
    row("AGC rescale к 4000", bench(agc), per_chunk_all=False)

    # --- 3. Silero VAD
    def vad():
        for c in chunks:
            eng._ww_vad_speech(c)
    row("Silero VAD (_ww_vad_speech)", bench(vad))

    # --- 4a. openWakeWord predict, split: fbank vs MLP
    mdl = eng.oww_model.models[eng.oww_model_inference_lock and 0] \
        if hasattr(eng, "oww_model_inference_lock") else None
    # openWakeWord Model keeps one inner model per path
    inner = list(eng.oww_model.models.values())[0]

    def predict_only():
        for c in live:
            eng.check_wakeword(c, 0.30, "x", True)
    row("openWakeWord predict (fbank+MLP целиком)",
        bench(predict_only), per_chunk_all=False)

    # fbank alone: openWakeWord caches per-hop audio, so feed a long buffer and
    # let it slide — that is what the per-chunk predict actually pays.
    def fbank_only():
        eng.oww_model.melspec(raw[:16000 * 20].astype(np.int16))
    try:
        row("  из них: melspectrogram (20s буфер)", bench(fbank_only),
            per_chunk_all=False)
    except Exception as e:  # API differs by version
        print(f"  (melspectrogram отдельно не измерить: {e})")

    def mlp_only():
        fb = eng.oww_model.fbanks[::2][:, :16]
        for _ in range(16):
            inner.run(inner.output_names, {"input_1": fb})
    try:
        row("  из них: MLP на 16 фреймов", bench(mlp_only),
            per_chunk_all=False)
    except Exception as e:
        print(f"  (MLP отдельно не измерить: {e})")

    # --- 5. echo correlation ring (44 offsets x 2560 samples)
    ring = np.random.default_rng(0).integers(
        -2000, 2000, 16000 * 60).astype(np.float32)
    rlen = len(ring)
    total = rlen
    probe = live[0].astype(np.float32) if live else np.zeros(CHUNK, np.float32)
    nn = len(probe)
    xn = float(np.linalg.norm(probe))

    def echo():
        best = 0.0
        for d in range(2, 46):
            ds = d * 16000
            start = total - ds - nn
            if start < 0:
                continue
            i0 = int(start) % rlen
            seg = (ring[i0:i0 + nn] if i0 + nn <= rlen
                   else np.concatenate([ring[i0:], ring[:i0 + nn - rlen]]))
            sn = float(np.linalg.norm(seg))
            if sn < 1e-3:
                continue
            corr = float(np.dot(probe, seg) / (xn * sn))
            best = max(best, corr)
        return best
    row("_is_echo: 44 смещения x 2560", bench(echo))

    print(f"\n{6.25:.2f} чанка/с на камеру; 8 камер = {6.25*8:.0f} чанков/с")


if __name__ == "__main__":
    main()