"""Feature extraction that matches the RUNTIME, not the training script.

The two previous training runs both passed an offline acceptance check and
still fired on live TV. The cause was not the model: it was the features.
`train_wake_head.py:features_for()` normalised the way a 1.28 s WINDOW, while
`camera_client._vad_process` normalises EVERY 160 ms chunk to peak 4000 before
the model ever sees it. Different amplitude dynamics in training vs production,
so the offline score said 0.785 while the live score said 0.895.

This module is the single place that decides what a feature vector is, and it
replays the production path exactly:

    raw 16 kHz int16
      -> 2560-sample (160 ms) chunks            [camera_client _drain_vad_buf]
      -> skip if raw peak < 600                [the meaningless-audio floor]
      -> scale to peak 4000 (both directions)   [_WW_TARGET_PEAK for livingroom]
      -> AudioFeatures streaming mel + CNN      [openwakeword, 80 ms hop]
      -> get_features(16) = 1.28 s x 96         [the head's input]

Two rules make it trustworthy:

  1. the chunk loop is shared with `scripts/eval_runtime_path.py`, so training
     data and the acceptance replay cannot drift apart;
  2. positives are cut out of the SAME normalised stream, so a positive window
     and a negative window differ only in content — never in level.

Do not reintroduce per-window normalisation. That is the bug this replaces.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNK = 2560        # 160 ms — camera_client._drain_vad_buf's carve size
AGC_TARGET = 4000   # livingroom's _WW_TARGET_PEAK
PEAK_FLOOR = 600    # below this the runtime does not call the model at all
N_FRAMES = 16       # head input: 16 embedding frames of 96 dims = 1.28 s


def runtime_chunks(raw_int16, target=AGC_TARGET, floor=PEAK_FLOOR):
    """Yield (normalised_chunk, raw_peak) exactly as the live path does.

    Chunks under the peak floor are SKIPPED, not yielded: in production they
    never reach the model, so training on them would teach the head about a
    signal it will never be asked about (and the AGC would inflate pure noise
    into something that looks like speech).
    """
    n = len(raw_int16) // CHUNK
    for i in range(n):
        c = raw_int16[i * CHUNK:(i + 1) * CHUNK]
        peak = int(np.max(np.abs(c.astype(np.int32))))
        if peak < floor:
            continue
        yield (np.clip(c.astype(np.float32) * (target / peak),
                       -32768, 32767).astype(np.int16), peak)


class FeatureExtractor:
    """Streaming feature extractor shared across files.

    One AudioFeatures instance for the whole run — constructing it loads two ONNX
    sessions (melspectrogram + embedding CNN) and doing that per burst turned a
    seconds-long job into minutes. Streaming state is reset between files
    instead, so one burst cannot leak context into the next.
    """

    def __init__(self, reset_stream=True):
        from openwakeword.utils import AudioFeatures
        self.af = AudioFeatures(ncpu=1)
        self.af.feature_buffer_max_len = 400
        self.reset_stream = reset_stream
        self.reset()

    def reset(self):
        """Clear streaming state so one clip cannot leak context into the next.

        The blank fill is exactly N_FRAMES rows, not the 116 that
        `_get_embeddings(np.zeros(160000))` produces — 10 s of silence through
        the embedding CNN per clip cost 850 ms and bought nothing, since only
        the last 16 frames are ever read. 32000 samples yields exactly 16 rows in
        105 ms.

        `_real` then gates emission (see feed), because 16 blank rows are worse
        than no rows: the first window would be 15/16 silence.
        """
        af = self.af
        if self.reset_stream:
            af.raw_data_buffer.clear()
        else:
            # the numpy ring installed by engine._use_numpy_ring_buffer has no
            # clear(); zero the write cursor instead and keep the hot path
            af._ring_w = 0
            af._ring_n = 0
        af.melspectrogram_buffer = np.ones((76, 32))
        af.accumulated_samples = 0
        af.feature_buffer = af._get_embeddings(
            np.zeros(N_FRAMES * 2000, dtype=np.int16))

    def feed(self, chunk_int16):
        """One 160 ms chunk -> one [1,16,96] vector, or None while the window is
        still filling.

        Warmup is counted in EMBEDDING ROWS, not chunks. A 160 ms chunk yields
        TWO rows (openWakeWord advances the mel by 8 frames per 1280 samples, so
        accumulated_samples=2560 produces two embeddings), so counting chunks
        against N_FRAMES=16 demanded 8 s of audio for a 1.28 s window and
        silently returned NOTHING for every short clip — the whole acceptance
        set came out empty before this was fixed.
        """
        af = self.af
        af._streaming_features(chunk_int16)
        real_rows = af.feature_buffer.shape[0] - N_FRAMES
        if real_rows < N_FRAMES:
            return None
        return af.get_features(N_FRAMES)

    def vectors(self, raw_int16, target=AGC_TARGET, floor=PEAK_FLOOR):
        """-> list of 1536-dim vectors, one per 80 ms hop, runtime-normalised."""
        self.reset()
        out = []
        for chunk, _peak in runtime_chunks(raw_int16, target, floor):
            f = self.feed(chunk)
            if f is not None:
                out.append(np.asarray(f, dtype=np.float32).reshape(-1))
        return out


def load_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"{path}: {w.getframerate()} Hz"
        assert w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def read_labels(path):
    """labels.tsv -> [(name, text)], skipping malformed lines."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            name, text = line.rstrip("\n").split("\t", 1)
            rows.append((name, text))
    return rows


def is_positive(text):
    """Whisper heard the wake word in this burst -> this is a wake positive.

    Token-based: any whitespace-separated token starting with «комп» counts, so
    «компьютер», «компьют» and a trailing period all match, while a word like
    «компактно» would not (the old bare `"комп" in text` matched that one).

    KNOWN LIMITATION, recorded rather than papered over: this cannot tell
    «компьютер, включи свет» (a real wake) from «компьютерная программа» (the
    user talking about computers, not to us). No transcript in the current set
    contains the second case — all 36 positives are the bare word — so this does
    not bite yet. If a future labelling run produces long sentences containing
    «комп», check those bursts by hand before they enter training.
    """
    t = text.lower().replace("ё", "е")
    return any(tok.startswith("комп") for tok in t.split())