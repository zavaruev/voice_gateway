"""Why does a 0.9994-CV head still fire on the TV? Hunt the shortcut, don't retrain.

The pattern is now established across three attempts: clean CV, clean offline
acceptance, then false fires in production. The offline acceptance is fed
feature vectors produced by the training code, so it cannot catch a shortcut
that lives in the AUDIO but not in the feature space — a shortcut that only
becomes visible when you re-derive features from a continuous stream.

This script looks for class-conditional structure along axes a model could
exploit, using the runtime feature extractor on real audio:

  * does the score track the chunk's crest factor / RMS?  (loud speech = wake?)
  * which SPECRAL region carries the separation?  (mean of the 96 embedding dims)
  * how does the score evolve over a burst, and does a 1.28 s window even have
    room for the word?

Run inside the image:
    python3 scripts/why_it_fires.py --model /data/lr_v3.onnx \\
        --dataset /data/ds_v2.npz --ambient /data/ambience.raw
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNK = 2560
N_FRAMES = 16


def stats_for(vec):
    """Per-frame stats of one feature vector: (16, 96) -> scalars."""
    f = np.asarray(vec, dtype=np.float32).reshape(N_FRAMES, 96)
    fr = np.sqrt((f ** 2).mean(axis=1))          # per-frame magnitude
    return dict(
        mag=float(np.sqrt((f ** 2).mean())),
        peak_frame=float(fr.max()),
        frame_std=float(fr.std()),
        dyn=float(fr.max() / (fr.mean() + 1e-9)),
        low=float(f[:, :32].mean()),
        mid=float(f[:, 32:64].mean()),
        high=float(f[:, 64:].mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ambient", required=True)
    ap.add_argument("--bursts", default="")
    ap.add_argument("--labels", default="")
    args = ap.parse_args()

    import onnxruntime as ort
    from wake_features import FeatureExtractor, is_positive, load_wav, read_labels

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name

    def score(v):
        return float(sess.run(None, {nm: np.asarray(v, np.float32)
                                     .reshape(1, N_FRAMES, 96)})[0]
                     .reshape(-1)[0])

    d = np.load(args.dataset, allow_pickle=True)
    X, y, meta = d["X"], d["y"], d["meta"]
    ps = np.asarray([score(v) for v in X])

    print("=" * 74)
    print("1. ЧТО РАЗДЕЛЯЕТ КЛАССЫ ПО ФИЧАМ")
    print("=" * 74)
    keys = ["mag", "peak_frame", "frame_std", "dyn", "low", "mid", "high"]
    rows = {}
    for k in keys:
        f = np.asarray([stats_for(v)[k] for v in X[::7]], dtype=np.float32)
        a, b = f[y[::7] == 1], f[y[::7] == 0]
        sep = abs(np.median(a) - np.median(b)) / (f.std() + 1e-9)
        rows[k] = sep
        print(f" {k:11s} pos {np.median(a):8.3f}  neg {np.median(b):8.3f}  "
              f"разделение {sep:5.2f} sigma")
    print(f"\n САМЫЙ СИЛЬНЫЙ КАНДИДАТ: "
          f"{max(rows, key=rows.get)} ({rows[max(rows, key=rows.get)]:.2f} sigma)")

    print()
    print("=" * 74)
    print("2. SCORE ПРОТИВ ЭТИХ ОСЕЙ (где именно модель ошибается)")
    print("=" * 74)
    feats = [stats_for(v) for v in X[::7]]
    sp = ps[::7]
    lab = y[::7]
    for k in keys:
        f = np.asarray([t[k] for t in feats])
        for lbl, m in ((f"score у Позитивов", lab == 1), (f"score у НЕГАТИВов", lab == 0)):
            if f[m].std() < 1e-9:
                continue
            c = float(np.corrcoef(f[m], sp[m])[0, 1])
            print(f" {k:11s} corr со {lbl.lower():12s} = {c:+.3f}")

    print()
    print("=" * 74)
    print("3. КАК SCORE ВЕДЁТ СЕБЯ ВНУТРИ BURST-а (окно 1.28 с, слово ~0.5 с)")
    print("=" * 74)
    if args.bursts and args.labels:
        from wake_features import runtime_chunks
        fx = FeatureExtractor()
        rows_n = read_labels(args.labels)
        shown = {"pos": 0, "neg": 0}
        for name, text in rows_n:
            want = "pos" if is_positive(text) else "neg"
            if shown[want] >= 3:
                continue
            p = os.path.join(args.bursts, name)
            if not os.path.exists(p):
                continue
            x = load_wav(p)
            shown[want] += 1
            fx.reset()
            seq = []
            for ch, _ in runtime_chunks(x):
                f = fx.feed(ch)
                seq.append(None if f is None else score(f))
            vals = [v for v in seq if v is not None]
            head = "".join(" . " if v is None else
                           ("#" if v >= 0.5 else ("+" if v >= 0.2 else "."))
                           for v in seq)
            print(f" {want} {name:16s} {len(x)/16000:4.2f}s -> {head}")
        print("  (# >=0.5, + >=0.2, . <0.2, пробел = окно не заполнено)")

    print()
    print("=" * 74)
    print("4. ЖИВОЙ ГЕЙТ: где в реальном потоке набирается score")
    print("=" * 74)
    raw = np.frombuffer(open(args.ambient, "rb").read(), dtype=np.int16)
    fx = FeatureExtractor()
    fx.reset()
    seq = []
    for i in range(len(raw) // CHUNK):
        c = raw[i * CHUNK:(i + 1) * CHUNK]
        pk = int(np.max(np.abs(c.astype(np.int32))))
        if pk < 600:
            continue
        w = np.clip(c.astype(np.float32) * (4000 / pk), -32768,
                    32767).astype(np.int16)
        f = fx.feed(w)
        seq.append((None if f is None else score(f),
                    float(np.sqrt((c.astype(np.float64) ** 2).mean())),
                    float(pk / (np.sqrt((c.astype(np.float64) ** 2).mean())
                                + 1e-6))))
    ok = [(s, r, cf) for s, r, cf in seq if s is not None]
    if ok:
        s = np.asarray([t[0] for t in ok])
        r = np.asarray([t[1] for t in ok])
        cf = np.asarray([t[2] for t in ok])
        print(f" окон {len(ok)}, score макс {s.max():.4f}, "
              f"p99.9 {np.percentile(s,99.9):.4f}")
        print(f" corr(score, rms)      = {np.corrcoef(s, r)[0,1]:+.3f}")
        print(f" corr(score, crest)    = {np.corrcoef(s, cf)[0,1]:+.3f}")
        hi = s >= 0.85
        if hi.any():
            print(f" при score>=0.85: rms медиана {np.median(r[hi]):.0f} "
                  f"(все окна: {np.median(r):.0f}), "
                  f"crest медиана {np.median(cf[hi]):.2f} "
                  f"(все: {np.median(cf):.2f})")


if __name__ == "__main__":
    main()