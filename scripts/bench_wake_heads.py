"""Screen wake heads on real room audio: noise rejection AND CPU cost.

The livingroom needs 8 cameras on a weak box, so the operating point is a
trade-off, not just accuracy: a head that needs a 5 ms scoring pass is
affordable, one that needs 40 ms is not, at 8 rooms x ~6 chunks/s.

Run inside the built image (needs `av`/onnxruntime and a 16 kHz mono s16le
capture on disk):

    python3 bench_wake_heads.py --noise /data/ambience.raw --pos /data/wake.raw
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
from engine import LocalAudioEngine  # noqa: E402

CHUNK = 2560
AGC_TARGET = 4000
PEAK_FLOOR = 600


def score_capture(model_path, raw, ncpu=1):
    """Score a capture exactly as camera_client would: per-chunk AGC, the 600
    floor, then check_wakeword. Returns (scores, ms_per_chunk)."""
    eng = LocalAudioEngine(vad_threshold=0.03)
    eng.initialize_models(model_path)
    out = []
    t0 = time.perf_counter()
    n = 0
    for i in range(len(raw) // CHUNK):
        c = raw[i * CHUNK:(i + 1) * CHUNK]
        pk = int(np.max(np.abs(c.astype(np.int32))))
        if pk < PEAK_FLOOR:
            out.append(0.0)
            continue
        w = np.clip(c.astype(np.float32) * (AGC_TARGET / pk),
                    -32768, 32767).astype(np.int16)
        eng.check_wakeword(w, 0.30, "bench", True)
        out.append(eng.last_score)
        n += 1
    dt = (time.perf_counter() - t0) / max(n, 1)
    return np.asarray(out), dt


def hits(scores, thresh, need=2, window=3):
    rec, n = [], 0
    for s in scores:
        if s >= thresh:
            rec.append(s)
        else:
            rec = []
        if len(rec) > window:
            rec = rec[-window:]
        if len(rec) >= need:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True, help="TV / ambience capture")
    ap.add_argument("--pos", default="", help="capture of the wake phrase")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--thresholds", default="0.30,0.60,0.85")
    args = ap.parse_args()

    noise = np.frombuffer(open(args.noise, "rb").read(), dtype=np.int16)
    pos = (np.frombuffer(open(args.pos, "rb").read(), dtype=np.int16)
           if args.pos else None)

    models = []
    for pat in args.models:
        models.extend(sorted(glob.glob(pat)))
    if not models:
        import openwakeword
        d = os.path.join(os.path.dirname(openwakeword.__file__),
                         "resources", "models")
        models = sorted(glob.glob(os.path.join(d, "*.onnx")))

    n_hours = len(noise) / 16000 / 3600
    print(f"шум: {len(noise)/16000:.0f}s ({n_hours*60:.1f} мин)   "
          f"позитив: {len(pos)/16000 if pos is not None else 0:.0f}s\n")
    hdr = f"{'модель':34s} {'мс/чанк':>8} {'на 8 камер':>11}"
    for t in args.thresholds.split(","):
        hdr += f" | шум≥{t}: ложн/час"
    if pos is not None:
        hdr += " | попаданий"
    print(hdr)

    for m in models:
        ns, ms = score_capture(m, noise)
        row = f"{os.path.basename(m)[:33]:34s} {ms*1000:7.1f} {ms*1000*6*8/1000:10.1f}%"
        for t in (float(x) for x in args.thresholds.split(",")):
            h = hits(ns, t)
            row += f" | {h/n_hours:11.1f}" if n_hours else " | ?"
        if pos is not None:
            ps, _ = score_capture(m, pos)
            hit = sum(1 for t in (float(x) for x in args.thresholds.split(","))
                      if hits(ps, t))
            row += f" | {hit}/{len(args.thresholds.split(','))}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
