"""Train a room-specific openWakeWord head on labelled livingroom recordings.

Pipeline mirrors the runtime exactly so the learned weights see the same
features they will see in production:

    audio -> AudioFeatures (melspectrogram.onnx -> embedding_model.onnx)
          -> 1280-sample (80 ms) windows of 16x96 = 1536 features
          -> sklearn MLP -> hand-built ONNX with input [1,16,96]

The runtime loads a model with input [1,16,96] and a single [1,1] output, so
the exported graph must match that signature exactly — no torch in the image,
so the graph is assembled directly with onnx.helper.

Labelling is NOT hand-made: scripts/record_bursts.py cuts utterances and
Whisper writes labels.tsv. A burst counts as positive only when Whisper heard
"комп..." in it. That keeps noise out of the positive class, which is the
failure mode that turns a wake model into a noise detector.
"""
import argparse
import os
import sys
import wave

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from openwakeword.utils import AudioFeatures
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

WINDOW = 1280    # 80 ms at 16 kHz - one raw frame fed to _streaming_features
N_FRAMES = 16    # embedding frames per prediction == 1.28 s of context


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, path
        assert w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


_AF = None


def _af():
    """One AudioFeatures for the whole run.

    Constructing it loads two ONNX sessions (melspectrogram + embedding); doing
    that per burst made the job take minutes instead of seconds. The streaming
    state is reset between files instead — see _reset.
    """
    global _AF
    if _AF is None:
        _AF = AudioFeatures(ncpu=1)
        _AF.feature_buffer_max_len = 400
    return _AF


def _reset(af):
    af.raw_data_buffer.clear()
    af.melspectrogram_buffer = np.ones((76, 32))
    af.accumulated_samples = 0
    af.feature_buffer = af._get_embeddings(np.zeros(160000).astype(np.int16))


def features_for(x):
    """-> list of 1536-dim feature vectors (16 embedding frames x 96 dims).

    The runtime calls AudioFeatures.get_features(16), i.e. every prediction
    sees the last 16 embedding frames = 1.28 s of audio, not a single 80 ms
    frame. Training has to reproduce that grouping or the classifier would be
    fitted to a context length it never sees in production.
    """
    af = _af()
    _reset(af)
    # 1 s of silence either side so the mel window is full at burst start and
    # the tail is flushed at the end.
    buf = np.concatenate([np.zeros(16000, dtype=np.int16), x,
                          np.zeros(16000, dtype=np.int16)])
    embeds = []
    for i in range(0, len(buf) - WINDOW + 1, WINDOW):
        af._streaming_features(buf[i:i + WINDOW].astype(np.int16))
        embeds.append(np.asarray(af.feature_buffer[-1]).reshape(-1))
    E = np.asarray(embeds, dtype=np.float32)
    if E.shape[0] < N_FRAMES:
        return []
    # stride 1 frame: predictions are made every 80 ms in the live path too
    return [E[i:i + N_FRAMES].reshape(-1) for i in range(E.shape[0] - N_FRAMES + 1)]


def augment(x, rng):
    """Mild room-style augmentation. Deliberately conservative: the target is
    THIS room, so we simulate distance and level, not other rooms."""
    out = []
    for gain in (0.25, 0.5, 0.8, 1.25, 1.8):
        y = np.clip(x.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        # low-frequency hum, as in the unamplified room
        n = len(y)
        t = np.arange(n) / 16000.0
        hum = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t))
        y = np.clip(y.astype(np.float32) + hum * 25.0, -32768, 32767).astype(np.int16)
        out.append(y)
    return out


def build_onnx(clf, scaler, path):
    """sklearn MLP -> a plain 2-layer MLP -> sigmoid, input [1,16,96]."""
    ws = []
    for coef, intercept in zip(clf.coefs_, clf.intercepts_):
        # sklearn coef_ is already (n_in, n_out) = MatMul's (K, N)
        ws.append(np.asarray(coef, dtype=np.float32))
        ws.append(np.asarray(intercept, dtype=np.float32))

    nodes, inits = [], []
    x = "X"

    # Standardiser folded into the first layer: Xn = (X - mean) / scale.
    # Order matters: subtract first, THEN divide. Doing Div-then-Sub computes
    # X/scale - mean, which is a different transform - it produced plausible
    # but wrong outputs (0.019 instead of 0.848 on the probe) rather than an
    # obvious failure, so the sklearn-vs-ONNX check below is what catches it.
    # mean/scale must be shaped [1,16,96]: a flat (1536,) tensor cannot
    # broadcast against the model's [1,16,96] input.
    shape = (1, N_FRAMES, 96)
    nodes.append(helper.make_node("Sub", [x, "mean"], ["Xc"]))
    inits.append(numpy_helper.from_array(
        scaler.mean_.astype(np.float32).reshape(shape), "mean"))
    nodes.append(helper.make_node("Div", ["Xc", "scale"], ["Xn3"]))
    inits.append(numpy_helper.from_array(
        scaler.scale_.astype(np.float32).reshape(shape), "scale"))
    # MatMul needs 2D: collapse [1,16,96] -> [1,1536] before the first layer.
    nodes.append(helper.make_node("Reshape", ["Xn3", "flat"], ["Xn"]))
    inits.append(numpy_helper.from_array(
        np.asarray([1, N_FRAMES * 96], dtype=np.int64), "flat"))

    nodes.append(helper.make_node("MatMul", ["Xn", "W0"], ["H0"]))
    inits.append(numpy_helper.from_array(ws[0], "W0"))
    nodes.append(helper.make_node("Add", ["H0", "B0"], ["H0b"]))
    inits.append(numpy_helper.from_array(ws[1], "B0"))
    nodes.append(helper.make_node("Relu", ["H0b"], ["A0"]))

    nodes.append(helper.make_node("MatMul", ["A0", "W1"], ["H1"]))
    inits.append(numpy_helper.from_array(ws[2], "W1"))
    nodes.append(helper.make_node("Add", ["H1", "B1"], ["H1b"]))
    inits.append(numpy_helper.from_array(ws[3], "B1"))
    nodes.append(helper.make_node("Sigmoid", ["H1b"], ["output"]))

    graph = helper.make_graph(
        nodes, "livingroom_wake",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 96])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        initializer=inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                          ir_version=8)
    m.doc_string = ("livingroom wake head, trained 03.10.2026 from 36 "
                    "Whisper-confirmed positives / 42 negatives")
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-negatives", type=int, default=260)
    ap.add_argument("--cache", default="",
                    help="npz path for cached features (speeds up reruns)")
    ap.add_argument("--extra-negatives", default="",
                    help="dir of ambience windows to add as negatives")
    ap.add_argument("--extra-labels", default="",
                    help="labels.tsv for --extra-negatives (same format)")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of the EXTRA negatives withheld from "
                         "training and used as the acceptance test only")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="operating point to check the acceptance set at")
    args = ap.parse_args()

    pos, neg = [], []
    for line in open(args.labels, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        path = os.path.join(args.bursts, name)
        if not os.path.exists(path):
            continue
        (pos if "комп" in text.lower() else neg).append((path, text))

    print(f"позитивов {len(pos)}, негативов {len(neg)}")
    rng = np.random.default_rng(7)

    # Feature extraction costs ~4 minutes (two ONNX models per 80 ms frame);
    # cache it so threshold/architecture iterations are seconds.
    # Ambience windows (TV / room tone). The FIRST burst set held only short
    # utterances, so the classifier had never seen sustained speech+music and
    # fired on it every ~25 s at 0.97-0.99. A slice is withheld from training
    # so the acceptance test is on audio the model has never fitted.
    amb_train, amb_hold = [], []
    if args.extra_negatives and args.extra_labels:
        rows = []
        for line in open(args.extra_labels, encoding="utf-8"):
            if "\t" not in line:
                continue
            name, text = line.rstrip("\n").split("\t", 1)
            p = os.path.join(args.extra_negatives, name)
            if not os.path.exists(p):
                continue
            if "комп" in text.lower():
                # never let the wake word into the negative class
                continue
            rows.append(p)
        n_hold = int(len(rows) * args.holdout)
        amb_hold = rows[:n_hold]          # earlier windows = acceptance set
        amb_train = rows[n_hold:]
        print(f"окон фона: {len(rows)} (обучение {len(amb_train)}, "
              f"проверка {len(amb_hold)})")

    cache = args.cache
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y = d["X"], d["y"]
        print(f"фичи из кэша: {X.shape}")
    else:
        X, y = [], []
        for path, _ in pos:
            v = features_for(load_wav(path))
            X += v
            y += [1] * len(v)
        print(f"  позитивных окон: {sum(y)}")

        neg_count = 0
        for path, text in neg:
            if neg_count >= args.max_negatives:
                break
            base = load_wav(path)
            # real speech negatives get level augmentation so the model cannot
            # simply learn "quiet == negative"
            variants = [base] + (augment(base, rng) if len(text.strip()) > 3 else [])
            for v in variants:
                feats = features_for(v)
                X += feats
                y += [0] * len(feats)
                neg_count += 1
        print(f"  отрицательных окон: {len(y) - sum(y)}")
        for p in amb_train:
            feats = features_for(load_wav(p))
            X += feats
            y += [0] * len(feats)
        print(f"  + окон фона в обучении: {len(y) - sum(y)}")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if cache:
            np.savez_compressed(cache, X=X, y=y)
            print(f"  кэш записан в {cache}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f"матрица {X.shape}, баланс {y.mean():.3f}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # early_stopping keeps this to seconds; max_iter=1200 without it took
    # >10 min for the 4-fold CV on 14.8k x 1536 and the extra iterations
    # changed nothing (validation plateaued long before).
    clf = MLPClassifier(hidden_layer_sizes=(48,), max_iter=300,
                        early_stopping=True, n_iter_no_change=10,
                        validation_fraction=0.15,
                        random_state=0, alpha=1e-3)
    # 1024 positives against ~18.5k negative windows is a 13:1 imbalance. A
    # wake model must not be taught "anything is negative", so the positive
    # class carries the inverse-frequency weight; sample_weight also reaches
    # cross_val_score, so the CV number reflects the weighted objective.
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    sample_weight = np.where(y == 1, len(y) / (2 * max(n_pos, 1)),
                             len(y) / (2 * max(n_neg, 1))).astype(np.float32)
    print(f"веса классов: positive={sample_weight[y == 1][0]:.3f} "
          f"negative={sample_weight[y == 0][0]:.4f}")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc",
                             params={"sample_weight": sample_weight})
    print(f"CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}  {scores}")

    clf.fit(Xs, y, sample_weight=sample_weight)
    path = build_onnx(clf, scaler, args.out)
    print(f"сохранено {path} ({os.path.getsize(path)} байт)")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # the graph declares a fixed batch of 1 (that is what the runtime feeds),
    # so compare one sample at a time
    idx = [int(i) for i in np.linspace(0, len(X) - 1, 5)]
    ref = clf.predict_proba(scaler.transform(X[idx]))[:, 1]
    got = [float(sess.run(None, {"X": X[i].reshape(1, N_FRAMES, 96)
                                 .astype(np.float32)})[0].reshape(-1)[0])
           for i in idx]
    print(f"проверка ONNX против sklearn: ref={np.round(ref, 5)}")
    print(f"                              onnx={np.round(got, 5)}")
    if not np.allclose(ref, got, atol=1e-4):
        print("ВНИМАНИЕ: ONNX не совпадает со sklearn")
        return 1

    # ---- acceptance: audio the model has never been fitted on ----
    if amb_hold:
        print(f"\nПРИЁМКА на {len(amb_hold)} отложенных окнах фона "
              f"(модель их не видела):")
        sess2 = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        nm = sess2.get_inputs()[0].name
        worst, fired, total = 0.0, 0, 0
        for p in amb_hold:
            for vec in features_for(load_wav(p)):
                r = float(sess2.run(None, {nm: vec.reshape(1, N_FRAMES, 96)
                                           .astype(np.float32)})[0].reshape(-1)[0])
                worst = max(worst, r)
                total += 1
                if r >= args.threshold:
                    fired += 1
        print(f"  порог {args.threshold:.2f}: ложных {fired} из {total} окон")
        print(f"  максимум score на отложенном фоне: {worst:.4f}")
        if fired:
            print("  ПРИЁМКА НЕ ПРОЙДЕНА — на фоне есть срабатывания")
            return 1
        print("  ПРИЁМКА ПРОЙДЕНА")
    return 0


if __name__ == "__main__":
    sys.exit(main())
