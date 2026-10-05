"""Replay a real room capture through the PRODUCTION pause detector.

The field log only ever printed the numbers at the moment an utterance ENDED
(`via pause (rms=... ref=... floor=... run=6 speech=11)`) or, when it did not
fire, nothing at all but `via cap 7s`. So a detector that never fires is
completely opaque: you cannot tell "the reference never dropped" from "the
speech floor swallowed every frame" without a trace. This prints one.

Feeds 160 ms frames computed exactly as `_vad_process` does
(`rms = sqrt(mean(square(int16))) / 32768`) into the real `_PauseEndpoint`,
under the same gate the live code uses: the endpoint only sees audio while an
utterance is open AND a wake window is active, so the harness opens one at the
first frame above `--speech-rms` and closes it when the cap would have.

Usage in the image (has numpy + camera_client):
    python3 scripts/replay_pause_endpoint.py capture.raw [--speech-rms 0.02]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp/vg_src")

from camera_client import _PauseEndpoint  # noqa: E402

FRAME = 16000 // 5  # 160 ms, the gateway's chunk size


def frames(raw: bytes):
    n = len(raw) // 2
    a = np.frombuffer(raw[: n * 2], dtype=np.int16)
    for i in range(0, len(a) - FRAME + 1, FRAME):
        b = a[i : i + FRAME].astype(np.float64)
        yield float(np.sqrt(np.mean(np.square(b)))) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--speech-rms", type=float, default=0.02,
                    help="level that opens the simulated wake window")
    ap.add_argument("--cap-s", type=float, default=7.0)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    raw = open(args.capture, "rb").read()
    cap_frames = int(args.cap_s / 0.16)
    ep = _PauseEndpoint()
    print(f"{len(raw) // 2 / 16000:.1f}s capture, cap {args.cap_s}s "
          f"({cap_frames} frames), ratio={ep.ratio} run_frames={ep.run_frames} "
          f"min_speech={ep.min_speech_frames} ref_frames={ep.ref_frames}")

    # Find speech bursts: contiguous runs above the open threshold.
    vals = list(frames(raw))
    print(f"{len(vals)} frames\n")
    open_win = False
    opened_at = None
    best = None
    for i, rms in enumerate(vals):
        if not open_win and rms >= args.speech_rms:
            open_win, opened_at = True, i
            ep.reset()
        if not open_win:
            continue
        verdict, detail = ep.feed(rms)
        if opened_at is not None:
            best = (opened_at, i, detail)
        print(f"  t={i * 0.16:5.2f}s rms={rms:.4f} {detail} -> {verdict}")
        if verdict == "end" or (i - opened_at) >= cap_frames:
            took = i - opened_at
            how = "PAUSE" if verdict == "end" else "CAP"
            print(f"  --- {how} after {took * 0.16:.2f}s "
                  f"({opened_at * 0.16:.2f}s..{i * 0.16:.2f}s) "
                  f"{best[2] if best else ''}")
            open_win, opened_at = False
            if i > args.max_frames:
                break


if __name__ == "__main__":
    main()
