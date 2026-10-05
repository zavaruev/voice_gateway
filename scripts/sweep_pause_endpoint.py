"""Compare pause-endpointing variants on a REAL capture, offline.

The field log can only tell you whether the shipped detector fired
(`via pause (...)`) or not (`via cap 7s`). It cannot tell you why. This replays
one real room capture through several detectors, so the choice is made from the
room's own audio instead of from an argument.

Measured 05.10.2026 on the living room, 21:37, 90 s capture: speech arrives at
rms 0.0248 and the room floor sits at 0.010-0.013. The shipped detector's
reference is a trailing 12-frame window, so it collapses from 0.0248 to 0.0123
within a second and silence (0.011) is never below 0.55 x 0.0123 = 0.0068 —
`run` stayed 0 for the whole 6.88 s utterance and it ended on the cap. Its speech
floor was 0.0072, BELOW the 0.011 background, so every quiet frame counted as
speech (`speech=41`).

Usage in the image:
    python3 scripts/sweep_pause_endpoint.py capture.raw [--min-ratio 0.4] ...
"""
import argparse

import numpy as np

FRAME = 16000 // 5  # 160 ms


def frame_rms(raw: bytes):
    n = len(raw) // 2
    a = np.frombuffer(raw[: n * 2], dtype=np.int16)
    for i in range(0, len(a) - FRAME + 1, FRAME):
        b = a[i : i + FRAME].astype(np.float64)
        yield float(np.sqrt(np.mean(np.square(b)))) / 32768.0


class Shipped:
    """The detector as deployed: trailing-window reference, frozen anchor floor."""

    name = "shipped (trailing ref)"

    def __init__(self, ratio=0.55, run_frames=6, min_speech=5, ref_frames=12,
                 anchor_frames=4, anchor_frac=0.5):
        self.ratio, self.run_frames, self.min_speech = ratio, run_frames, min_speech
        self.ref_frames, self.anchor_frames = ref_frames, anchor_frames
        self.anchor_frac = anchor_frac
        self.reset()

    def reset(self):
        self._ref, self._utt, self._anchor = [], [], 0.0
        self._run, self._speech = 0, 0

    def feed(self, rms):
        self._ref.append(rms)
        if len(self._ref) > self.ref_frames:
            self._ref.pop(0)
        ref = float(np.percentile(self._ref, 80))
        if self._anchor == 0.0 and len(self._utt) < self.anchor_frames:
            self._utt.append(rms)
            if len(self._utt) >= self.anchor_frames:
                self._anchor = float(np.median(self._utt))
        floor = self._anchor * self.anchor_frac
        is_pause = ref > 0.0 and rms < ref * self.ratio
        if is_pause:
            self._run += 1
        else:
            self._run = 0
            if floor > 0.0 and rms >= floor:
                self._speech += 1
        if self._run >= self.run_frames and self._speech >= self.min_speech:
            self.reset()
            return "end"
        return "pause" if is_pause else "speech"


class Frozen:
    """Utterance-scoped reference: high percentile of the SPEECH seen so far.

    The windowed reference decays to the room floor within ~1 s of silence,
    which is why a 7 s utterance of a 1.5 s command never found a pause. Here the
    reference is built from frames already judged speech and can only RISE, so a
    pause stays a pause for as long as it lasts. Rising is also the safe
    direction: a door slam raises it and the utterance falls back to the cap —
    slower, never chopped.

    Classification is then self-consistent: pause and speech are complements of
    one threshold, so `min_speech_frames` counts frames that are NOT a pause
    instead of comparing against a second, independent floor.
    """

    name = "frozen speech ref"

    def __init__(self, ratio=0.55, run_frames=6, min_speech=5, ref_frames=12):
        self.ratio, self.run_frames, self.min_speech = ratio, run_frames, min_speech
        self.reset()

    def reset(self):
        self._win, self._spk = [], []
        self._ref, self._run, self._speech = 0.0, 0, 0

    def feed(self, rms):
        self._win.append(rms)
        if len(self._win) > 12:
            self._win.pop(0)
        if self._ref == 0.0:
            # Bootstrap on the windowed rule, exactly as the shipped detector
            # does, until there is enough speech to stand on its own.
            boot = float(np.percentile(self._win, 80))
            is_pause = boot > 0.0 and rms < boot * self.ratio
            if not is_pause:
                self._spk.append(rms)
                self._speech += 1
            self._run = self._run + 1 if is_pause else 0
            if self._speech >= self.min_speech:
                self._ref = float(np.percentile(self._spk, 80))
        else:
            is_pause = rms < self._ref * self.ratio
            if is_pause:
                self._run += 1
            else:
                self._run = 0
                self._speech += 1
                self._ref = max(self._ref, rms)
        if self._run >= self.run_frames and self._speech >= self.min_speech:
            self.reset()
            return "end"
        return "pause" if is_pause else "speech"


def run(cls, vals, ratio, open_rms, cap_frames, **kw):
    """Simulate the live gate: the endpoint only sees audio inside a wake window."""
    ep = cls(ratio=ratio, **kw)
    open_win = False
    opened = 0
    ends = []
    for i, rms in enumerate(vals):
        if not open_win and rms >= open_rms:
            open_win, opened = True, i
        if not open_win:
            continue
        verdict = ep.feed(rms)
        if verdict == "end" or (i - opened) >= cap_frames:
            ends.append(("pause" if verdict == "end" else "cap", i - opened))
            open_win = False
    return ends


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--open-rms", type=float, default=0.02)
    ap.add_argument("--cap-s", type=float, default=7.0)
    args = ap.parse_args()

    raw = open(args.capture, "rb").read()
    vals = list(frame_rms(raw))
    cap_frames = int(args.cap_s / 0.16)
    print(f"{len(vals) * 0.16:.1f}s, open_rms={args.open_rms}, "
          f"cap={args.cap_s}s\n")

    print(f"{'detector':<24}{'ratio':>6}{'ends':>6}{'pause':>7}{'cap':>6}"
          f"{'mean s':>9}  worst")
    for cls in (Shipped, Frozen):
        for ratio in (0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
            ends = run(cls, vals, ratio, args.open_rms, cap_frames)
            if not ends:
                print(f"{cls.name:<24}{ratio:>6.2f}{0:>6}{'-':>7}{'-':>6}"
                      f"{'-':>9}  (no window opened)")
                continue
            secs = [n * 0.16 for _, n in ends]
            n_pause = sum(1 for k, _ in ends if k == "pause")
            print(f"{cls.name:<24}{ratio:>6.2f}{len(ends):>6}{n_pause:>7}"
                  f"{len(ends) - n_pause:>6}{np.mean(secs):>9.2f}  {max(secs):.2f}s")


if __name__ == "__main__":
    main()
