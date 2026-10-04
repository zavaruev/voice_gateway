"""Cut continuous ambience into fixed 1.28 s windows as negatives.

The burst set contains short utterances (0.3-2 s) with 1 s of silence padded
either side, so the classifier never saw the room's SUSTAINED noise floor.
That is exactly what it fires on in production: the livingroom sits at
rms ~400 / peak ~1400 at audio.volume 100, and every ~25 s the model reaches
0.97-0.99 on it. Positive scores saturate at 1.000, so no threshold separates
the two — the only fix is showing the model what the real ambience looks like.

Emits windows, not utterance bursts: the model's unit is a 16-frame (1.28 s)
context, so negatives must be cut the same way.
"""
import argparse
import os
import wave

import numpy as np

SR = 16000
CTX = 2048 * 5  # 1.28 s = 16 embedding frames of 80 ms, stride 0.64 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="mono s16le 16 kHz capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=float, default=0.5,
                    help="window overlap; 0.5 => a 50%%-overlapped sweep")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    x = np.frombuffer(open(args.raw, "rb").read(), dtype=np.int16)
    os.makedirs(args.out, exist_ok=True)
    win = int(CTX / max(args.stride, 0.01)) if args.stride < 1 else int(CTX)
    step = max(win - int(win * args.stride), 1)

    n = 0
    for i in range(0, len(x) - win + 1, step):
        seg = x[i:i + win]
        # level-normalise like the runtime's bidirectional AGC does, so the
        # model sees the same scale it will see live
        pk = int(np.max(np.abs(seg.astype(np.int32))))
        if pk < 100:
            continue  # digital silence teaches nothing
        seg = np.clip(seg.astype(np.float32) * (4000 / pk), -32768, 32767).astype(np.int16)
        with wave.open(os.path.join(args.out, f"amb_{n:04d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(seg.tobytes())
        n += 1
        if n >= args.limit:
            break

    print(f"окон: {n} по {win/SR:.2f}s (шаг {step/SR:.2f}s) из {len(x)/SR:.0f}s")
    if n:
        print(f"первое окно rms={np.sqrt(np.mean(x[:win].astype(np.float64)**2)):.0f}")


if __name__ == "__main__":
    main()
