"""Reproduce the production feed path, because the offline probe cannot see it.

The user said «компьютер» twice and the room did not wake. This reproduces that
exactly: the wake word is embedded in CONTINUOUS room audio and fed to the
matcher the way `_vad_process` does — begin() once, feed() chunk after chunk,
no utterance boundary, no flush().

Why that differs from the offline probe
    The probe decoded a FIXED 2.5 s window, so both the trigger and the
    confirmation saw audio containing the word. In production the room's Silero
    VAD never reports silence (0 `speech=False` in 15 live minutes at
    audio.volume=100), so there is no utterance boundary at all: the matcher
    begins once, hits `max_secs`, resets, and begins again. The confirmation
    window is then anchored to the START of that arbitrary 3 s window — which is
    mostly television, and the 0.6 s word often falls outside the first 2 s.

This script measures that failure directly instead of assuming it, by planting
a real recorded wake burst at a known offset in real room audio and counting
hits as a function of that offset.
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "/app")
# NOTE: do NOT prepend this script's directory. A copy of vosk_wake.py left in /tmp
# would then shadow /app's and the reproduction would silently test stale code —
# which it did, reporting 0/7 for a build that fires correctly.


def load_wav(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def mix_snr(fg, bg, snr_db):
    fg = fg.astype(np.float32)
    need = len(fg)
    if len(bg) < need:
        bg = np.tile(bg.astype(np.float32), int(np.ceil(need / len(bg))))[:need]
    else:
        s = int(np.random.default_rng(0).integers(0, len(bg) - need))
        bg = bg[s:s + need].astype(np.float32)
    r = np.sqrt(np.mean(fg ** 2)) + 1e-6
    rb = np.sqrt(np.mean(bg ** 2)) + 1e-6
    y = fg + bg * (r / (rb * 10 ** (snr_db / 20.0)))
    pk = float(np.max(np.abs(y)))
    return (y * (32767 / pk) if pk > 32767 else y).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="config/vosk-model-small-ru-0.22")
    ap.add_argument("--room", default="/tmp/live.raw")
    ap.add_argument("--bursts", nargs="+", required=True)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--gains", default="1.0,0.5")
    ap.add_argument("--offsets", type=int, default=8)
    args = ap.parse_args()

    from vosk_wake import VoskWakeMatcher

    room = np.frombuffer(open(args.room, "rb").read(), dtype=np.int16)
    words = [load_wav(p) for p in args.bursts]
    print(f"комната {len(room)/16000:.0f}s (rms "
          f"{np.sqrt((room.astype(float)**2).mean()):.0f}), "
          f"{len(words)} слов, SNR {args.snr:g} дБ\n")

    gains = [float(g) for g in args.gains.split(",")]

    for gain in gains:
        print("=" * 68)
        print(f"подача КАК В БОЮ: begin() один раз, feed() потоком, без flush()"
              f"   (усиление слова {gain:g})")
        print("=" * 68)
        hits = offs = 0
        step = max(1600, (len(room) - 6000) // max(args.offsets, 1))
        for off in range(0, len(room) - 6000, step):
            stream = room.copy()
            w = words[offs % len(words)]
            w = (np.clip(w.astype(np.float32) * gain, -32768,
                         32767)).astype(np.int16)
            w = mix_snr(w, room, args.snr)
            end = off + len(w)      # plant the WHOLE burst: truncating it to
            # 4000 samples left a 0.25 s fragment that no decoder can finalise,
            # and the reproduction reported 0 % for a build that fires.
            if end > len(stream):
                break
            stream[off:end] = w[:end - off]
            offs += 1

            m = VoskWakeMatcher(args.model)
            m.begin()
            fired = False
            for i in range(0, len(stream), 3200):
                if m.feed(stream[i:i + 3200].tobytes()):
                    fired = True
                    break
            if not fired and m.flush():        # flush only helps if it ever runs
                fired = True
            hits += fired
            if fired:
                print(f"    слово на {off/16000:4.1f}c -> ДА "
                      f"trig='{m.last_trigger_text}' conf='{m.last_text}'")
        print(f"\n  ИТОГО {hits}/{offs} срабатываний "
              f"({100*hits/max(offs,1):.0f}%)\n")


if __name__ == "__main__":
    main()