"""Score both wake heads on the SAME burst set and report the operating point.

The point is not "AUC is high" (the training CV already said 0.996) but the
per-utterance hit rate at the threshold the gateway actually uses for this
room (0.30), with the 2-of-3 debounce applied — that is what decides whether
the room wakes reliably.

Also reports how far the positives sit above 0.30, because a model that fires
on 1 of 40 attempts usually has the positives sitting just under the line.
"""
import argparse
import os
import re
import wave

import numpy as np
import onnxruntime as ort
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wake_head import N_FRAMES, WINDOW, features_for, load_wav


def scores_for(model_path, wav_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for vec in features_for(load_wav(wav_path)):
        r = sess.run(None, {name: vec.reshape(1, N_FRAMES, 96).astype(np.float32)})
        out.append(float(r[0].reshape(-1)[0]))
    return np.asarray(out)


def debounce_hit(scores, thresh, need=2, window=3):
    """True if `need` of the last `window` scores are >= thresh — the same
    rule _vad_process uses, so a lone spike does not count as a wake."""
    recents = []
    for s in scores:
        if s >= thresh:
            recents.append(s)
        else:
            recents = []
        if len(recents) > window:
            recents = recents[-window:]
        if len(recents) >= need:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--thresh", type=float, default=0.30)
    args = ap.parse_args()

    pos, neg = [], []
    for line in open(args.labels, encoding="utf-8"):
        if "\t" not in line:
            continue
        name, text = line.rstrip("\n").split("\t", 1)
        p = os.path.join(args.bursts, name)
        if os.path.exists(p):
            (pos if "комп" in text.lower() else neg).append(p)

    for tag, path in (("старая (библиотечная)", args.old),
                      ("новая (гостиная)", args.new)):
        print(f"\n=== {tag}: {os.path.basename(path)}")
        # score every burst ONCE, then evaluate all thresholds against the
        # cached scores - re-extracting features per threshold made this take
        # minutes per model
        pos_s = [scores_for(path, p) for p in pos]
        neg_s = [scores_for(path, p) for p in neg]
        m = np.asarray([float(s.max()) for s in pos_s])
        hits = sum(1 for s in pos_s if debounce_hit(s, args.thresh))
        fp = sum(1 for s in neg_s if debounce_hit(s, args.thresh))
        print(f"  попаданий на «компьютер»: {hits}/{len(pos)} "
              f"({hits/max(len(pos),1)*100:.0f}%)")
        print(f"  ложных на другой речи/шуме: {fp}/{len(neg)}")
        print(f"  максимум score по позитивам: медиана {np.median(m):.3f}, "
              f"мин {m.min():.3f}, макс {m.max():.3f}")
        print(f"  ниже порога {args.thresh}: {(m < args.thresh).sum()}/{len(m)}")
        for thr in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
            h = sum(1 for s in pos_s if debounce_hit(s, thr))
            f = sum(1 for s in neg_s if debounce_hit(s, thr))
            print(f"    порог {thr:.2f}: попаданий {h:2d}/{len(pos)}  ложных {f:2d}/{len(neg)}")


if __name__ == "__main__":
    main()
