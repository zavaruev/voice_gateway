"""Candidate pause detectors, scored on the room's OWN audio.

Three, same input: 160 ms rms frames from a real capture.

  shipped  trailing 12-frame 80th percentile. On the living room this reference
           is poisoned twice: the attention pip is ~60x the room floor when the
           utterance opens, and once the pip leaves the 1.9 s window the
           reference decays TO the floor, after which silence is never below 55 %
           of the reference because it IS the reference. Measured 05.10.2026:
           4 of 4 real commands ended on the 7 s cap.

  frozen   reference from the utterance's own speech frames, may only rise.
           Still seeded by the pip, and a reference that can only rise stays
           stuck high for the rest of the utterance.

  floor    no speech reference at all. A room noise floor tracked CONTINUOUSLY
           — fast down, very slow up — and a frame is a pause when it sits below
           `floor * mult`. The floor is a property of the ROOM, not of the
           utterance, so it is fed on every frame and survives `reset()`; that is
           also what fixes the seeding problem, since the attention pip can no
           longer define it. One threshold means pause and speech are exact
           complements and cannot disagree about what "quiet" means.

Ground truth is `truth`: the last frame at or above 45 % of the utterance's own
90th percentile — a property of the audio, not of any detector.

Usage in the image:
    python3 scripts/compare_pause_detectors.py /tmp/utt/*.wav --seed capture.raw
"""
import argparse
import glob
import sys
import wave

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp/vg_src")

from camera_client import _PauseEndpoint  # noqa: E402


def wav_frames(path):
    with wave.open(path, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    step = 16000 * 160 // 1000
    return _rms(a, step)


def raw_frames(path):
    b = open(path, "rb").read()
    a = np.frombuffer(b[: (len(b) // 2) * 2], dtype=np.int16)
    return _rms(a, 16000 * 160 // 1000)


def _rms(a, step):
    return [
        float(np.sqrt(np.mean(np.square(a[i : i + step].astype(np.float64)))))
        / 32768.0
        for i in range(0, len(a) - step + 1, step)
    ]


class NoiseFloor:
    """Room noise floor: fast down, very slow up. Room-scoped, not utterance."""

    def __init__(self, fast_down=0.4, slow_up=0.003, seed_frames=12):
        self.fast_down, self.slow_up, self.seed_frames = fast_down, slow_up, seed_frames
        self.value = 0.0
        self._seed: list[float] = []

    def feed(self, rms: float) -> None:
        if self.value == 0.0:
            self._seed.append(rms)
            if len(self._seed) >= self.seed_frames:
                # Seed with the LOW end of the window, so a pip in the first
                # frames cannot become the floor.
                self.value = float(min(self._seed))
            return
        if rms < self.value:
            self.value += (rms - self.value) * self.fast_down
        else:
            self.value += (rms - self.value) * self.slow_up


class FloorDetector:
    """Pause = below `mult` x a continuously tracked room floor."""

    def __init__(self, noise: NoiseFloor, mult=2.2, run_frames=6, min_speech=5):
        self.noise, self.mult = noise, mult
        self.run_frames, self.min_speech = run_frames, min_speech
        self.reset()

    def reset(self):
        self._run = 0
        self._speech = 0

    def feed(self, rms):
        thresh = self.noise.value * self.mult
        is_pause = self.noise.value > 0.0 and rms < thresh
        if is_pause:
            self._run += 1
        else:
            self._run = 0
            self._speech += 1
        detail = f"floor={self.noise.value:.4f} thresh={thresh:.4f} run={self._run} speech={self._speech}"
        if self._run >= self.run_frames and self._speech >= self.min_speech:
            self.reset()
            return "end", detail
        return ("pause" if is_pause else "speech"), detail


def fire_shipped(vals, ratio=0.55, run_frames=6):
    ep = _PauseEndpoint(ratio=ratio, run_frames=run_frames)
    for i, v in enumerate(vals):
        verdict, _ = ep.feed(v)
        if verdict == "end":
            return i
    return None


def truth(vals):
    hi = float(np.percentile(vals, 90))
    last = 0
    for i, v in enumerate(vals):
        if v >= hi * 0.45:
            last = i
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--seed", help="capture used to establish the room floor")
    ap.add_argument("--mults", default="2.0,2.2,2.5,3.0")
    args = ap.parse_args()

    wavs, raws = [], []
    for g in args.globs:
        found = sorted(glob.glob(g))
        (wavs if g.endswith(".wav") else raws).extend(found)
    mults = [float(x) for x in args.mults.split(",")]

    seed = raw_frames(args.seed) if args.seed else []

    def make_floor():
        nf = NoiseFloor()
        for v in seed:
            nf.feed(v)
        return nf

    print(f"room floor from {args.seed or 'none'}: "
          f"{make_floor().value:.4f}\n")
    print("=== real commands (from /tmp/utterances) ===")
    hdr = f"{'file':<22}{'truth':>8}{'shipped':>10}" + "".join(
        f"{'fl ' + str(m):>9}" for m in mults)
    print(hdr)
    agg = {m: [] for m in mults}
    for p in wavs:
        vals = wav_frames(p)
        t = truth(vals) * 0.16
        s = fire_shipped(vals)
        s_txt = "cap" if s is None else f"{s * 0.16:.2f}s"
        cells = []
        for m in mults:
            nf = make_floor()
            nf.value = nf.value or 0.01
            ep = FloorDetector(nf, mult=m)
            # Prime the noise tracker with this utterance's own frames too, the
            # way the live loop would have seen them before the wake.
            for v in vals[:12]:
                nf.feed(v)
            fired = None
            for i, v in enumerate(vals):
                verdict, _ = ep.feed(v)
                if verdict == "end":
                    fired = i
                    break
            cells.append(fired)
            agg[m].append(((fired if fired is not None else len(vals)) * 0.16) - t)
        cell_txt = [("cap" if c is None else f"{c * 0.16:.2f}s") for c in cells]
        print(f"{p.rsplit('/', 1)[-1]:<22}{t:>7.2f}s{s_txt:>10}"
              + "".join(f"{c:>9}" for c in cell_txt))
    print(f"{'overshoot vs truth':<22}{'-':>8}{'-':>10}"
          + "".join(f"{np.mean(agg[m]):>8.2f}s" for m in mults))

    if raws:
        for p in raws:
            vals = raw_frames(p)
            print(f"\n=== ambient {p.rsplit('/', 1)[-1]} ({len(vals) * 0.16:.0f}s) ===")
            cap = int(7.0 / 0.16)
            for m in mults:
                nf = NoiseFloor()
                ep = FloorDetector(nf, mult=m)
                open_win, opened, ends = False, 0, []
                for i, v in enumerate(vals):
                    nf.feed(v)
                    if not open_win and v >= 0.02:
                        open_win, opened = True, i
                    if not open_win:
                        continue
                    verdict, _ = ep.feed(v)
                    if verdict == "end" or (i - opened) >= cap:
                        ends.append("pause" if verdict == "end" else "cap")
                        open_win = False
                print(f"  mult={m}: {len(ends)} windows, "
                      f"{ends.count('pause')} pause / {ends.count('cap')} cap")


if __name__ == "__main__":
    main()
