"""The numpy ring buffer that engine.py installs over openWakeWord's audio path.

Separate from test_engine.py on purpose: that module permanently replaces
ort.InferenceSession with a MagicMock and never restores it, so a test that
needs a real ONNX session would silently skip there. This file does NOT mock
onnxruntime — it loads the actual melspectrogram/embedding/wake ONNX graphs,
which is the whole point: the claim under test is bit-identical scores.

Docker-only (needs silero_vad.onnx + config/*.onnx), like the rest of the
image test set.
"""
import os

import numpy as np
import pytest

from engine import _ring_buffer_tail, _use_numpy_ring_buffer

WAKE = "config/computer_20260706_130638.onnx"
EMB = "config/embedding_model.onnx"


def _models_present():
    return all(os.path.exists(p) for p in (WAKE, EMB, "silero_vad.onnx"))


requires_models = pytest.mark.skipif(
    not _models_present(),
    reason="real openWakeWord ONNX graphs not present (run inside the image)",
)


def _chunks(n=40, size=2560):
    """Deterministic stand-in for a mic capture: modulated tone plus noise, so
    the scores actually move instead of sitting at the model's floor."""
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        t = np.arange(size) / 16000.0
        f = 180.0 + 40.0 * np.sin(i * 0.3)
        sig = (np.sin(2 * np.pi * f * t) * 6000).astype(np.float32)
        sig += (rng.standard_normal(size) * 900).astype(np.float32)
        if i % 7 == 3:
            sig *= 4.0
        out.append(np.clip(sig, -32768, 32767).astype(np.int16))
    return out


def _scores(chunks):
    from openwakeword import Model
    m = Model(wakeword_model_paths=[WAKE], embedding_onnx_model_path=EMB)
    return m, [float(max(m.predict(c).values())) for c in chunks]


@requires_models
def test_ring_buffer_scores_identically():
    """The ring must be bit-identical to openWakeWord's deque.

    It trades 3.8 ms of per-call garbage (rebuilding 160 000 Python ints to
    use 3040) for a numpy view. If even one sample landed in the wrong slot,
    every deployed threshold would shift silently — a wake model that drifts
    is worse than a slow one.
    """
    chunks = _chunks()

    stock, stock_scores = _scores(chunks)
    assert not getattr(stock.preprocessor, "_numpy_ring", False)

    from openwakeword import Model
    patched = Model(wakeword_model_paths=[WAKE], embedding_onnx_model_path=EMB)
    _use_numpy_ring_buffer(patched)
    assert patched.preprocessor._numpy_ring
    patched_scores = [float(max(patched.predict(c).values())) for c in chunks]

    assert patched_scores == stock_scores, (
        "ring buffer changed scores; max delta "
        f"{max(abs(a - b) for a, b in zip(patched_scores, stock_scores))}"
    )
    # guard against a vacuous pass
    assert max(stock_scores) > 0.0


@requires_models
def test_ring_buffer_is_idempotent():
    """initialize_models() may be called more than once (reload, tests); a
    second apply must not stack rings on top of each other."""
    from openwakeword import Model
    patched = Model(wakeword_model_paths=[WAKE], embedding_onnx_model_path=EMB)
    _use_numpy_ring_buffer(patched)
    ring = patched.preprocessor._ring
    _use_numpy_ring_buffer(patched)
    assert patched.preprocessor._ring is ring


@requires_models
def test_ring_buffer_wraparound_keeps_newest_in_order():
    """The ring is capped at 10 s, so it wraps on any long session. After
    wrapping, the mel window must still be the NEWEST n+480 samples, in
    chronological order — a reversed or stale window would score silence."""
    from openwakeword import Model
    m = Model(wakeword_model_paths=[WAKE], embedding_onnx_model_path=EMB)
    _use_numpy_ring_buffer(m)
    p = m.preprocessor
    assert p.raw_data_buffer.maxlen == 160_000

    rng = np.random.default_rng(3)
    for _ in range(300):  # ~24 s of 1280-sample writes: wraps the 10 s ring
        block = rng.integers(-3000, 3000, 1280).astype(np.int16)
        p._buffer_raw_data(block)
        window = _ring_buffer_tail(
            p._ring, p._ring_w, p._ring_n, p.raw_data_buffer.maxlen,
            1280 + 160 * 3,
        )
        assert window.dtype == np.int16
        assert np.array_equal(window[-1280:], block)


@requires_models
def test_ring_buffer_rejects_tiny_frames_like_upstream():
    """openWakeWord raises below 400 samples; keep that contract or a silent
    caller could starve the mel hop counter forever."""
    from openwakeword import Model
    m = Model(wakeword_model_paths=[WAKE], embedding_onnx_model_path=EMB)
    _use_numpy_ring_buffer(m)
    with pytest.raises(ValueError):
        m.preprocessor._buffer_raw_data(np.zeros(100, dtype=np.int16))


def test_ring_buffer_maths_without_onnx():
    """Wrap/order logic, no ONNX needed: a fake preprocessor is enough, so the
    arithmetic is covered on the host suite too."""
    from engine import _use_numpy_ring_buffer as apply

    class FakePreprocessor:
        """Stand-in for AudioFeatures: captures the mel window instead of
        running the graph, so the wrap arithmetic is testable without ONNX."""

        def __init__(self):
            self.raw_data_buffer = type("D", (), {"maxlen": 1600})()
            self.melspectrogram_buffer = np.zeros((76, 32))
            self.melspectrogram_max_len = 970
            self.captured = []

        def _get_melspectrogram(self, x):
            self.captured.append(np.array(x, copy=True))
            return np.zeros((1, 32))

    class Fake:
        def __init__(self):
            self.preprocessor = FakePreprocessor()

    f = Fake()
    apply(f)
    p = f.preprocessor

    rng = np.random.default_rng(11)
    for _ in range(200):  # 100x the 1600-sample ring
        block = rng.integers(-3000, 3000, 1280).astype(np.int16)
        p._buffer_raw_data(block)
        p._streaming_melspectrogram(1280)

    got = p.captured[-1]
    assert got.dtype == np.int16
    # the window is n_samples+480, but never more than the 1600-sample ring holds
    assert len(got) == p.raw_data_buffer.maxlen
    assert np.array_equal(got[-1280:], block)

    # an empty ring must hand the mel model an empty array, as the deque did
    assert len(
        _ring_buffer_tail(f.preprocessor._ring, 0, 0,
                          f.preprocessor.raw_data_buffer.maxlen, 1760)
    ) == 0