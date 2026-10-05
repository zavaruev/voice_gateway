"""Which floor can follow a room that gets loud, without a sentence raising it?

The failure this scores is measured, not hypothetical. In the living room with the
television on, the saved follow-up utterance looks like this (raw rms, 160 ms):

    7068 14950 7792 4185 3880 3125 2766 2836 2978 2951 ... 2963   (44 frames)
    |--- the user, 0.8 s ----|------ television, flat, 6+ s ------|

Every one of those frames is above the detector's threshold, so `run` never
reached 6 and the utterance ran the full 7 s cap — and a 0.8 s command diluted
into 6 s of television is what Whisper fails to transcribe.

The reported floor was 0.0103 (337 raw) while the room actually sat at
2800-3300 raw. So the floor was stale by 8x, and it was stale BY CONSTRUCTION:
the fast-down/slow-up tracker only accepts frames it already considers noise, so
a room that gets loud can never raise its own reference.

Candidates:

  fastdown  the deployed tracker. Cannot rise into a loud room (the bug).
  window20  rolling 20th percentile over a long window (94 frames = 15 s).
            Speech is a minority of frames in any 15 s, so the low percentile
            stays at the room; a television running for minutes occupies all of
            them and IS the room. A two-frame attention pip is invisible to a
            20th percentile over 94 frames, which is what defeated the trailing
            window in the first place.
  window35  same, less headroom against speech.

Ground truth is the last frame belonging to the user, read off the envelope above
rather than off any detector.
"""
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp/vg_src")

STEP = 16000 * 160 // 1000


def envelope(path):
    import wave

    with wave.open(path, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return [
        float(np.sqrt(np.mean(np.square(a[i : i + STEP].astype(np.float64)))))
        for i in range(0, len(a) - STEP + 1, STEP)
    ]


class FastDown:
    """The deployed tracker: only frames it already calls noise may move it."""

    name = "fastdown (deployed)"

    def __init__(self, mult=2.5, seed=12, fast_down=0.4, slow_up=0.003):
        self.mult, self.seed = mult, seed
        self.fast_down, self.slow_up = fast_down, slow_up
        self.value = 0.0
        self._seed = []

    def feed(self, rms):
        if self.value == 0.0:
            self._seed.append(rms)
            if len(self._seed) >= self.seed:
                self.value = float(min(self._seed))
            return
        if rms >= self.value * self.mult:
            return
        if rms < self.value:
            self.value += (rms - self.value) * self.fast_down
        else:
            self.value += (rms - self.value) * self.slow_up


class WindowPercentile:
    name = "window20"

    def __init__(self, pct=20, frames=94, mult=2.0):
        self.pct, self.frames, self.mult = pct, frames, mult
        self.value = 0.0
        self._w: list[float] = []

    def feed(self, rms):
        self._w.append(rms)
        if len(self._w) > self.frames:
            self._w.pop(0)
        if len(self._w) >= 8:
            self.value = float(np.percentile(self._w, self.pct))

    def __repr__(self):
        return f"{self.name} pct={self.pct} n={self.frames} mult={self.mult} value={self.value:.0f}"


def fire(floors, vals, mult, run_frames=6, min_speech=5, prefill=0.0):
    """Returns the frame index at which the pause is confirmed, or None.

    `prefill` seeds the floor's window with the room's steady level, which is what
    the live tracker would already hold after the television has been on for a
    while — replaying an utterance in isolation would otherwise blame the
    detector for starting cold.
    """
    nf = floors(mult)
    for _ in range(nf.frames if hasattr(nf, "frames") else 12):
        nf.feed(prefill)
    run = speech = 0
    for i, rms in enumerate(vals):
        nf.feed(rms)
        thresh = nf.value * mult
        is_pause = nf.value > 0 and rms < thresh
        if is_pause:
            run += 1
        else:
            run, speech = 0, speech + 1
        if run >= run_frames and speech >= min_speech:
            return i
    return None


def main() -> None:
    import glob

    # (path, label, last user frame, room steady level for the prefill)
    cases = [
        ("/tmp/utt2/u_1791229446.wav", "follow-up (TV)", 4, 2950),
        ("/tmp/utt2/u_1791229531.wav", "follow-up (TV)", 4, 1200),
        ("/tmp/utt2/u_1791229434.wav", "wake + command", 1, 400),
        ("/tmp/utt2/u_1791229519.wav", "wake + command", 1, 400),
    ]
    print(f"{'case':<20}{'user ends':>10}{'deployed':>12}"
          f"{'win20 x1.6':>12}{'win20 x2.0':>12}{'win20 x2.5':>12}")
    for p, label, truth, prefill in cases:
        vals = envelope(p)
        row = []
        d = fire(FastDown, vals, 2.5, prefill=prefill)
        row.append("cap" if d is None else f"{d * 0.16:.2f}s")
        for mult in (1.6, 2.0, 2.5):
            i = fire(
                lambda m, pct=20: WindowPercentile(pct=pct), vals, mult,
                prefill=prefill,
            )
            row.append("cap" if i is None else f"{i * 0.16:.2f}s")
        print(f"{label:<20}{truth * 0.16:>9.2f}s" + "".join(f"{c:>12}" for c in row))

    print("\nfloor each candidate settles on for the follow-up case:")
    vals = envelope(cases[0][0])
    for name, make, mult in (
        ("deployed", lambda m: FastDown(), 2.5),
        ("win20", lambda m: WindowPercentile(20, 94), 2.0),
    ):
        nf = make(mult)
        for _ in range(94):
            nf.feed(2950)
        for v in vals:
            nf.feed(v)
        print(f"  {name:<10} floor={nf.value:8.0f}  thresh={nf.value * mult:8.0f}"
              f"  room=2950  user={max(vals):.0f}")


if __name__ == "__main__":
    main()
