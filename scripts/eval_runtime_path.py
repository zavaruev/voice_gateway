"""Score audio through the EXACT runtime wake path, not an approximation.

This exists because the offline evaluation lied. The first two training runs
passed a held-out acceptance test (0 false out of 3900 ambience windows) and
still fired on the TV in production at 0.89. The cause was preprocessing: the
training features were normalised per 1.28 s window while camera_client
normalises every 80 ms chunk to peak 4000, so the model was fitted on
amplitude dynamics it never meets live.

So: feed raw 16 kHz mic audio through the same steps _vad_process runs —
per-chunk bidirectional AGC to 4000, the 600 raw-peak floor, the Silero gate,
then check_wakeword — and report hits / false fires with the 2-of-3 debounce.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The engine lives in the app root, and LocalAudioEngine.initialize_models()
# loads "silero_vad.onnx" by RELATIVE path — so both sys.path AND the working
# directory have to be the app root. Running this from /tmp died twice: first
# ModuleNotFoundError, then NO_SUCHFILE on silero_vad.onnx.
_APP = "/app" if os.path.isdir("/app") else os.getcwd()
sys.path.insert(0, _APP)
os.chdir(_APP)
from engine import LocalAudioEngine

CHUNK = 2560        # 160 ms, what _drain_vad_buf carves out
AGC_TARGET = 4000   # livingroom per-room target
PEAK_FLOOR = 600    # below this the model is not called at all


def runtime_scores(raw_int16, engine, threshold, target=AGC_TARGET):
    """Yield the score for every chunk the gateway would actually score."""
    out = []
    recent: list[float] = []
    for i in range(len(raw_int16) // CHUNK):
        s16 = raw_int16[i * CHUNK:(i + 1) * CHUNK]
        raw_peak = int(np.max(np.abs(s16.astype(np.int32))))
        if raw_peak < PEAK_FLOOR:
            recent = []
            out.append((float("nan"), raw_peak))
            continue
        s16_w = np.clip(
            s16.astype(np.float32) * (target / raw_peak), -32768, 32767
        ).astype(np.int16)
        vad_ok = engine._ww_vad_speech(s16)
        engine.check_wakeword(s16_w, threshold, "livingroom", vad_ok)
        sc = float(engine.last_score)
        out.append((sc, raw_peak))
    return out


def fires(scored, threshold, need=2, window=3, gap_s=2.0):
    """Same 2-of-3 sliding debounce _vad_process applies, counted in EVENTS.

    Counting debounce-satisfied CHUNKS overstates the rate by ~6x: a single
    one-second episode satisfies the debounce on every one of its ~6 chunks and
    gets reported as six false activations. A new event is only counted after
    `gap_s` seconds below threshold, which is also what the operator perceives as
    "it went off again".

    Returns (events, chunks) — quote the first, the second is for debugging.
    """
    recent: list[float] = []
    events, n_chunks = 0, 0
    last = -10 ** 9
    gap = int(gap_s * 16000 / CHUNK)
    for i, (sc, _peak) in enumerate(scored):
        if not np.isnan(sc) and sc >= threshold:
            recent.append(sc)
        else:
            recent = []
        if len(recent) > window:
            recent = recent[-window:]
        if len(recent) >= need:
            n_chunks += 1
            if i - last > gap:
                events += 1
            last = i
    return events, n_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True, help="mono s16le 16 kHz capture")
    ap.add_argument("--label", default="")
    ap.add_argument("--thresholds", default="0.70,0.80,0.85,0.90,0.95,0.99")
    ap.add_argument("--seconds", type=int, default=0)
    args = ap.parse_args()

    x = np.frombuffer(open(args.raw, "rb").read(), dtype=np.int16)
    if args.seconds:
        x = x[:args.seconds * 16000]
    dur = len(x) / 16000
    print(f"{args.label or args.raw}: {dur:.0f}s")
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    print(f"  rms {rms:.0f}")

    engine = LocalAudioEngine(vad_threshold=0.03)
    engine.initialize_models(args.model)

    for thr in (float(t) for t in args.thresholds.split(",")):
        scored = runtime_scores(x, engine, thr)
        vals = np.asarray([s for s, _ in scored if not np.isnan(s)])
        events, chunks = fires(scored, thr)
        per_hour = events / dur * 3600 if dur else 0
        print(f"  порог {thr:.2f}: СОБЫТИЙ {events:3d} "
              f"({per_hour:7.1f}/час)  чанков с дебаунсом {chunks:4d}  "
              f"макс score {vals.max() if vals.size else 0:.4f}  "
              f"p99.9 {np.percentile(vals, 99.9) if vals.size else 0:.4f}")


if __name__ == "__main__":
    main()
