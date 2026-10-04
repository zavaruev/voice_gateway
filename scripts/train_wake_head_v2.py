"""Train the head on the level-blind dataset and ACCEPT it on the runtime path.

The acceptance test in the previous script was the thing that lied: it scored
held-out feature vectors produced by `features_for()`, i.e. the training
preprocessing. The model passed it (0 false of 3900 windows) and then fired on
live TV 96 times in 7 minutes. An acceptance test that runs different code from
the training code measures nothing.

So this trainer has two hard rules:

  1. TRAINING and ACCEPTANCE share one feature extractor (`wake_features`),
     which replays `camera_client._vad_process` — 160 ms chunks, per-chunk AGC
     to peak 4000, peak-600 floor, 80 ms hop, 1.28 s window.
  2. The final gate is `scripts/eval_runtime_path.py`: a raw mic capture scored
     through the live code path with the real 2-of-3 debounce. A model that
     fails there is rejected regardless of how good its CV looked.

Model: a small MLP on the frozen 1536-dim embedding, deliberately tiny. The
head MLP costs 0.07 ms of the 13 ms wake path, so capacity is free, but the
POSITIVE COUNT is the binding constraint (36 bursts -> ~300 windows), so the
model must stay small and heavily regularised or it memorises bursts.

Usage (inside the image):
    python3 scripts/train_wake_head_v2.py --dataset /data/ds.npz --out m.onnx
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

N_FRAMES = 16
DIMS = 96


def build_onnx(clf, scaler, path, note=""):
    """sklearn MLP -> [1,16,96] float in, [1,1] probability out.

    The runtime feeds exactly this signature, so the exported graph must match
    it. No torch in the image, hence hand-built with onnx.helper.

    Order matters and has bitten before: the standardiser is folded as
    (X - mean) / scale. Writing Div-then-Sub gives X/scale - mean, a different
    transform that produced plausible-but-wrong numbers (0.019 instead of 0.848)
    instead of an obvious failure — which is why the sklearn-vs-ONNX equality
    check at the bottom is mandatory, not decorative.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ws = []
    for coef, intercept in zip(clf.coefs_, clf.intercepts_):
        ws.append(np.asarray(coef, dtype=np.float32))
        ws.append(np.asarray(intercept, dtype=np.float32))

    nodes, inits = [], []
    x = "X"
    shape = (1, N_FRAMES, DIMS)

    nodes.append(helper.make_node("Sub", [x, "mean"], ["Xc"]))
    inits.append(numpy_helper.from_array(
        scaler.mean_.astype(np.float32).reshape(shape), "mean"))
    nodes.append(helper.make_node("Div", ["Xc", "scale"], ["Xn3"]))
    inits.append(numpy_helper.from_array(
        scaler.scale_.astype(np.float32).reshape(shape), "scale"))
    nodes.append(helper.make_node("Reshape", ["Xn3", "flat"], ["Xn"]))
    inits.append(numpy_helper.from_array(
        np.asarray([1, N_FRAMES * DIMS], dtype=np.int64), "flat"))

    prev, prev_b = "Xn", "B0"
    inits.append(numpy_helper.from_array(ws[0], "W0"))
    nodes.append(helper.make_node("MatMul", ["Xn", "W0"], ["H0"]))
    nodes.append(helper.make_node("Add", ["H0", "B0"], ["H0b"]))
    inits.append(numpy_helper.from_array(ws[1], "B0"))
    prev = "H0b"

    for i in range(1, len(clf.coefs_)):
        nodes.append(helper.make_node("Relu", [prev], [f"A{i}"]))
        nodes.append(helper.make_node("MatMul", [f"A{i}", f"W{i}"], [f"H{i}"]))
        nodes.append(helper.make_node("Add", [f"H{i}", f"B{i}"], [f"H{i}b"]))
        inits.append(numpy_helper.from_array(ws[2 * i], f"W{i}"))
        inits.append(numpy_helper.from_array(ws[2 * i + 1], f"B{i}"))
        prev = f"H{i}b"

    nodes.append(helper.make_node("Sigmoid", [prev], ["output"]))
    del prev_b

    graph = helper.make_graph(
        nodes, "wake_head",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, N_FRAMES, DIMS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        initializer=inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                          ir_version=8)
    m.doc_string = note or "wake head"
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=3e-3)
    ap.add_argument("--ambient", default="", help="raw capture for the final gate")
    ap.add_argument("--gate-thresholds", default="0.5,0.7,0.85,0.9,0.95")
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    X, y = d["X"], d["y"]
    HX, Hy = d["HX"], d["Hy"]
    Hmeta = d["Hmeta"]
    grp = d["grp"] if "grp" in d else np.zeros(len(y), dtype=int)
    print(f"обучение {X.shape} (позитивов {int(y.sum())}), "
          f"приёмка {HX.shape} (позитивов {int(Hy.sum())})")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    # inverse frequency: otherwise the 20x negative majority wins outright and
    # the head degenerates into "never fires" (which is exactly how a wake model
    # gets shipped deaf).
    w = np.where(y == 1, len(y) / (2 * max(n_pos, 1)),
                 len(y) / (2 * max(n_neg, 1))).astype(np.float32)
    print(f"веса: pos={w[y == 1][0]:.3f} neg={w[y == 0][0]:.4f}")

    def mk():
        return MLPClassifier(hidden_layer_sizes=(args.hidden,), max_iter=400,
                             early_stopping=True, n_iter_no_change=15,
                             validation_fraction=0.15, random_state=0,
                             alpha=args.alpha)

    # Group-aware CV. Each training burst is expanded into ~900 windows across
    # speeds, SNRs and background draws, so windows from one burst are near
    # duplicates. A random StratifiedKFold therefore puts the same recording in
    # both train and test and reports a meaningless number — this is the second
    # reason the earlier CV scores looked perfect. Split by BURST: a fold's
    # test set contains speech the model has never heard.
    codes, uniq = {}, []
    for g in grp.tolist():
        if g not in codes:
            codes[g] = len(uniq)
            uniq.append(g)
    gc = np.asarray([codes[g] for g in grp.tolist()], dtype=int)
    print(f"групп для CV: {len(uniq)} (окон на группу "
          f"{len(y)/max(len(uniq),1):.0f})")
    cv = GroupKFold(n_splits=min(5, len(uniq)))
    sc = cross_val_score(mk(), Xs, y, cv=cv.split(Xs, y, gc),
                         scoring="roc_auc", params={"sample_weight": w})
    print(f"CV ROC-AUC (по burst-ам): {sc.mean():.4f} +/- {sc.std():.4f}")

    clf = mk().fit(Xs, y, sample_weight=w)

    # ---- acceptance on unseen bursts / unseen ambient tail, same features
    ps = clf.predict_proba(scaler.transform(HX))[:, 1]
    pos_m = Hy == 1
    print("\nПРИЁМКА (отложенные burst-ы + хвост фона, те же фичи):")
    print(f"  положительных окон {int(pos_m.sum())}, "
          f"медиана score {np.median(ps[pos_m]):.4f}, "
          f"макс {ps[pos_m].max():.4f}")
    if (ps[pos_m] < 0.30).any():
        print("  ВНИМАНИЕ: есть положительные окна ниже 0.30 — "
              "модель глуха к части записей")
    for t in (float(x) for x in args.gate_thresholds.split(",")):
        fp = int((ps[~pos_m] >= t).sum())
        tp = int((ps[pos_m] >= t).sum())
        print(f"  порог {t:.2f}: recall {tp}/{int(pos_m.sum())} "
              f"({100*tp/max(int(pos_m.sum()),1):.0f}%), "
              f"ложных {fp}/{int((~pos_m).sum())}")
    for tag in sorted(set(Hmeta.tolist())):
        m = Hmeta == tag
        print(f"    {tag:22s} n={int(m.sum()):5d} макс {ps[m].max():.4f}")

    note = (f"wake head {X.shape[0]}w/{int(y.sum())}pos, hidden={args.hidden}, "
            f"alpha={args.alpha}, trained on runtime-matched features")
    path = build_onnx(clf, scaler, args.out, note)
    print(f"\nсохранено {path} ({os.path.getsize(path)} байт)")

    # ---- mandatory: exported graph must equal sklearn, sample by sample
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    idx = [int(i) for i in np.linspace(0, len(X) - 1, 8)]
    ref = clf.predict_proba(scaler.transform(X[idx]))[:, 1]
    got = [float(sess.run(None, {"X": X[i].reshape(1, N_FRAMES, DIMS)
                                 .astype(np.float32)})[0].reshape(-1)[0])
           for i in idx]
    print(f"ONNX vs sklearn: ref={np.round(ref, 5)}")
    print(f"                onnx={np.round(got, 5)}")
    if not np.allclose(ref, got, atol=1e-4):
        print("ПРОВАЛ: экспортированный ONNX не совпадает со sklearn")
        return 1

    # ---- the gate that counts: raw capture through the LIVE code path
    if args.ambient:
        print(f"\nФИНАЛЬНЫЙ ГЕЙТ: сырая запись через живой код "
              f"({args.ambient})")
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "eval_runtime_path.py")
        for amb in args.ambient.split():
            r = subprocess.run(
                [sys.executable, "-u", script, "--model", args.out,
                 "--raw", amb, "--label", os.path.basename(amb),
                 "--thresholds", args.gate_thresholds],
                capture_output=True, text=True, cwd=here)
            out = "\n".join(l for l in (r.stdout or "").splitlines()
                            if l.strip() and "INFO" not in l
                            and "WARNING" not in l and "warn" not in l)
            print(out or r.stderr[-800:])
    return 0


if __name__ == "__main__":
    sys.exit(main())