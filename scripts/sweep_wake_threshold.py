"""Threshold sweep against the held-out ambience set + the positive bursts.

The acceptance test is binary at one threshold, which hides the information
needed to choose the threshold: where the positives actually sit versus where
the false alarms reach. This prints both curves so the operating point can be
picked from measurements instead of guessed.
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wake_head import N_FRAMES, features_for, load_wav


def session(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def all_scores(sess, name, wavs):
    out = []
    for p in wavs:
        for vec in features_for(load_wav(p)):
            out.append(float(sess.run(None, {name: vec.reshape(1, N_FRAMES, 96)
                                             .astype(np.float32)})[0]
                              .reshape(-1)[0]))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos-bursts", required=True)
    ap.add_argument("--pos-labels", required=True)
    ap.add_argument("--amb-dir", required=True)
    ap.add_argument("--amb-count", type=int, default=150)
    args = ap.parse_args()

    pos = []
    for line in open(args.pos_labels, encoding="utf-8"):
        if "\t" not in line:
            continue
        n, t = line.rstrip("\n").split("\t", 1)
        p = os.path.join(args.pos_bursts, n)
        if os.path.exists(p) and "комп" in t.lower():
            pos.append(p)

    amb = sorted(os.path.join(args.amb_dir, f)
                 for f in os.listdir(args.amb_dir) if f.endswith(".wav"))
    amb = amb[:args.amb_count]
    print(f"позитивов {len(pos)}, окон фона {len(amb)}")

    sess, name = session(args.model)
    pos_scores = [all_scores(sess, name, [p]) for p in pos]
    amb_scores = all_scores(sess, name, amb)

    pm = np.asarray([s.max() for s in pos_scores])
    print(f"\nпозитивы: медиана максимума {np.median(pm):.3f}, "
          f"мин {pm.min():.3f}, 10-й перцентиль {np.percentile(pm,10):.3f}")
    print(f"фон: всего окон {len(amb_scores)}, "
          f"максимум {amb_scores.max():.3f}, "
          f"99.9-й перцентиль {np.percentile(amb_scores,99.9):.3f}")

    print(f"\n{'порог':>7} {'попаданий':>12} {'ложных на фоне':>16} {'окон/час':>10}")
    for thr in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99):
        hits = sum(1 for s in pos_scores if s.max() >= thr)
        # windows are 1.28 s with 50% overlap -> ~2 per second of audio
        fps = float((amb_scores >= thr).sum()) * 2.0
        mark = ""
        if fps < 0.02 and hits == len(pos):
            mark = "  <- можно ставить"
        print(f"{thr:7.2f} {hits:6d}/{len(pos):<5d} "
              f"{int((amb_scores >= thr).sum()):6d} из {len(amb_scores):<7d} "
              f"{fps:10.2f}{mark}")


if __name__ == "__main__":
    main()
