"""Where does a real command actually END, and when could a detector have said so?

`/tmp/utterances/*.wav` holds every utterance the gateway sent to Whisper, so the
room's own commands are already on disk — no need to ask anyone to speak again.
For each one this prints the 160 ms rms envelope and then asks two detectors when
they would have ended the utterance:

  * shipped — the trailing-window reference that is deployed;
  * frozen  — a reference built from the utterance's own speech frames that can
    only rise.

Measured 05.10.2026 18:32 UTC on the living room: «включи свет» and «выключи
свет» both ran the full 7 s cap, so the reply started ~7 s after the user stopped
talking. This says whether that is the detector's fault or the room's.

Usage in the image:
    python3 scripts/analyze_command_latency.py /tmp/utterances/*.wav
"""
import glob
import sys
import wave

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp/vg_src")

from camera_client import _PauseEndpoint  # noqa: E402

FRAME_MS = 160
SPEED = 1.0


def frames(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    step = int(16000 * FRAME_MS / 1000)
    out = []
    for i in range(0, len(a) - step + 1, step):
        b = a[i : i + step].astype(np.float64)
        out.append(float(np.sqrt(np.mean(np.square(b)))) / 32768.0)
    return out


class Frozen:
    """Reference from the utterance's own speech frames; may only rise."""

    def __init__(self, ratio=0.55, run_frames=6, min_speech=5):
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
            return True
        return False


def first_detector_fire(vals, ratio, kind, run_frames=6):
    ep = _PauseEndpoint(ratio=ratio, run_frames=run_frames) if kind == "shipped" \
        else Frozen(ratio=ratio, run_frames=run_frames)
    for i, rms in enumerate(vals):
        if kind == "shipped":
            v, _ = ep.feed(rms)
            if v == "end":
                return i
        else:
            if ep.feed(rms):
                return i
    return None


def energy_end(vals, ratio=0.45):
    """Ground truth from the envelope alone: last frame above ratio x the
    utterance's own 90th percentile. Crude, but it is a property of the audio,
    not of any detector."""
    hi = float(np.percentile(vals, 90))
    last = 0
    for i, v in enumerate(vals):
        if v >= hi * ratio:
            last = i
    return last, hi


def main() -> None:
    paths = []
    for a in sys.argv[1:]:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("no wav files given")
        return

    print(f"{'file':<22}{'dur':>6}{'last speech':>12}{'p90':>8}"
          f"{'shipped .55':>13}{'frozen .55':>12}{'frozen .60':>12}"
          f"{'frozen .65':>12}")
    for p in paths:
        vals = frames(p)
        last, hi = energy_end(vals)
        name = p.rsplit("/", 1)[-1]
        cells = []
        for kind, ratio in (("shipped", 0.55), ("frozen", 0.55),
                            ("frozen", 0.60), ("frozen", 0.65)):
            i = first_detector_fire(vals, ratio, kind)
            cells.append("cap 7s" if i is None else f"{i * 0.16:.2f}s")
        print(f"{name:<22}{len(vals) * 0.16:>5.1f}s{last * 0.16:>11.2f}s"
              f"{hi:>8.4f}{cells[0]:>13}{cells[1]:>12}{cells[2]:>12}{cells[3]:>12}")

    # Full envelope of the first file, so the numbers above can be checked.
    p = paths[0]
    vals = frames(p)
    print(f"\nenvelope of {p.rsplit('/', 1)[-1]} (160 ms frames):")
    for i in range(0, len(vals), 2):
        bar = "#" * int(vals[i] / 0.0015)
        print(f"  {i * 0.16:5.2f}s {vals[i]:.4f} {bar}")


if __name__ == "__main__":
    main()
