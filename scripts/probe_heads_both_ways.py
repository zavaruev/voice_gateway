"""Does any READY head hear the Russian «компьютер» AND ignore the TV?

Both halves matter and neither alone decides it:
  * the library heads are all 0.0 false/hour on the livingroom TV (measured),
    but a head that is deaf to the word is useless;
  * our room head hears the word at 1.000 but fires on the TV 96 times an hour.

So each candidate is scored BOTH ways on the same material: 36 Whisper-confirmed
positives (burst WAVs) and 420 s of TV ambience (raw capture), through the
runtime path (per-80ms-chunk AGC, 600 floor, 2-of-3 debounce).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import LocalAudioEngine  # noqa: E402

CHUNK = 2560
AGC_TARGET = 4000
PEAK_FLOOR = 600


def runtime_scores(raw, engine):
    out = []
    for i in range(len(raw) // CHUNK):
        c = raw[i * CHUNK:(i + 1) * CHUNK]
        pk = int(np.max(np.abs(c.astype(np.int32))))
        if pk < PEAK_FLOOR:
            out.append(0.0)
            continue
        w = np.clip(c.astype(np.float32) * (AGC_TARGET / pk),
                    -32768, 32767).astype(np.int16)
        engine.check_wakeword(w, 0.30, "x", True)
        out.append(float(engine.last_score))
    return np.asarray(out)


def fired(scores, thresh, need=2, window=3):
    rec, n = [], 0
    for s in scores:
        rec.append(s) if s >= thresh else rec.clear()
        if len(rec) > window:
            rec = rec[-window:]
        if len(rec) >= need:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ambience", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    pos_files = []
    for line in open(args.labels, encoding="utf-8"):
        if "\t" not in line:
            continue
        n, t = line.rstrip("\n").split("\t", 1)
        if "комп" not in t.lower():
            continue
        p = os.path.join(args.bursts, n)
        if os.path.exists(p):
            pos_files.append(p)
    amb = np.frombuffer(open(args.ambience, "rb").read(), dtype=np.int16)
    hours = len(amb) / 16000 / 3600
    print(f"позитивов {len(pos_files)}, ТВ {hours*60:.1f} мин\n")

    print(f"{'модель':30s} {'попаданий':>12} {'медиана':>8} {'макс':>7} "
          f"{'ложн/час ТВ':>12}")
    for m in args.models:
        eng = LocalAudioEngine(vad_threshold=0.03)
        eng.initialize_models(m)
        scores = []
        for p in pos_files:
            x = np.frombuffer(open(p, "rb").read(), dtype=np.int16)
            s = runtime_scores(x, eng)
            scores.append(s)
        hits = sum(1 for s in scores if fired(s, 0.30))
        med = float(np.median([s.max() for s in scores])) if scores else 0.0
        mx = float(max(s.max() for s in scores)) if scores else 0.0
        ns = runtime_scores(amb, eng)
        fa = fired(ns, 0.30) / hours if hours else 0.0
        print(f"{os.path.basename(m)[:29]:30s} {hits:6d}/{len(pos_files):<5d} "
              f"{med:8.3f} {mx:7.3f} {fa:12.1f}", flush=True)


if __name__ == "__main__":
    main()
