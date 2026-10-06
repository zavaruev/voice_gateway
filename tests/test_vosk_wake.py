"""Tests for vosk_wake.py — the decode-based wake word.

Docker-only like the other ONNX/vosk tests: the real model is 88 MB and only
present in the image. The pure parts (tolerant matching, the streaming protocol,
the state machine) are covered here without loading it, because those are where
the logic lives — a typo in the token matcher would otherwise only show up as a
silent wake word in a noisy room.

The design is a FREE (unconstrained) decoder, and the reason is the whole point
of this file's first test: the grammar-constrained variant had 100 % recall on
isolated bursts and produced ONE hypothesis across 25 s of live room speech. See
`vosk_wake.VoskWakeMatcher`'s docstring.

Numbers this was chosen for, measured on real material: 3/3 detections at 9 dB
SNR planted into live room audio, and 0 false accepts on 99 windows of held-out
television — against 229 false activations/hour for the acoustic head trained on
the same room.
"""
import contextlib
import json

import numpy as np
import pytest

# Deliberately NO sys.path.insert("/app") here. The other test files import their
# modules by bare name so pytest resolves them from the directory under test; this
# one originally pinned /app first, which made it exercise the module baked into
# the image rather than the copy being tested — and reported a NameError from a
# stale file while the source on disk was correct.
vosk_wake = pytest.importorskip("vosk_wake")


# --- the tolerant token matcher ------------------------------------------

def test_transcript_has_wake_matches_stem():
    """vosk spells the word inconsistently on short/noisy audio, so a prefix
    match on a token is required. Equality would report a working detector as
    broken."""
    assert vosk_wake.transcript_has_wake("компьютер")
    assert vosk_wake.transcript_has_wake("Компьютер включи свет")
    assert vosk_wake.transcript_has_wake("комп")                 # clipped
    assert vosk_wake.transcript_has_wake("ну компьютер")
    assert vosk_wake.transcript_has_wake("компьютер.")           # punctuation
    assert vosk_wake.transcript_has_wake("компЬЮТЕР")            # ё -> е


def test_transcript_has_wake_rejects_everything_else():
    """«компот» and «компания» are the reason this is an allowlist: a prefix
    match on "комп" admitted both, and each occurrence is a false activation."""
    for text in ("", "да", "не", "телевизор", "компания", "компот", "компьютерн",
                 "как дела", "подожди"):
        assert not vosk_wake.transcript_has_wake(text), text


def test_transcript_has_wake_handles_none():
    assert not vosk_wake.transcript_has_wake(None)


# --- streaming protocol --------------------------------------------------

class _FakeRec:
    """Minimal KaldiRecognizer stand-in with a scripted hypothesis sequence."""

    def __init__(self, model, sr, grammar=None):
        self.grammar = grammar
        self.seen = 0
        self.final_calls = 0
        self.hypotheses = list(getattr(_FakeRec, "script", ["да"]))

    def SetWords(self, _):
        pass

    def AcceptWaveform(self, data):
        self.seen += len(data)
        return False

    def PartialResult(self):
        h = self.hypotheses.pop(0) if self.hypotheses else ""
        return json.dumps({"partial": h})

    def FinalResult(self):
        self.final_calls += 1
        return json.dumps({"text": ""})


@contextlib.contextmanager
def fake_recognizer(script=("да",)):
    """Swap vosk.KaldiRecognizer for the duration of a test.

    Must wrap `begin()` too: constructing a real recogniser against the stub
    model raises before any stub of ours is in place.
    """
    _FakeRec.script = list(script)
    import vosk
    orig = getattr(vosk, "KaldiRecognizer", None)
    vosk.KaldiRecognizer = lambda model, sr, grammar=None: _FakeRec(
        model, sr, grammar)
    try:
        yield
    finally:
        if orig is not None:
            vosk.KaldiRecognizer = orig


@pytest.fixture
def matcher():
    """A VoskWakeMatcher whose model is a plain object."""
    m = vosk_wake.VoskWakeMatcher.__new__(vosk_wake.VoskWakeMatcher)
    m.model_path = "fake"
    m.max_secs = 1.0
    m._model = object()
    m.triggers = 0
    m.decodes = 0
    m.reset()
    return m


CHUNK = np.zeros(3200, dtype=np.int16).tobytes()   # 200 ms


def test_feed_is_noop_until_begin(matcher):
    """An utterance must be started explicitly on VAD onset: a fresh recogniser
    per window is what bounds the decoder's context in a room whose VAD never
    reports silence."""
    assert matcher.feed(CHUNK) is False
    assert matcher.triggers == 0


def test_matcher_uses_a_free_decoder(matcher):
    """Regression guard for the design decision: passing a grammar to
    KaldiRecognizer makes it reject the wake word on live room audio."""
    seen = {}

    class _GrammarSpy(_FakeRec):
        def __init__(self, model, sr, grammar=None):
            seen["grammar"] = grammar
            super().__init__(model, sr, grammar)

    with fake_recognizer(("да",)):
        import vosk
        orig = vosk.KaldiRecognizer
        vosk.KaldiRecognizer = _GrammarSpy
        try:
            matcher.begin()
            matcher.feed(CHUNK)
        finally:
            vosk.KaldiRecognizer = orig

    assert seen["grammar"] is None, (
        "a grammar-constrained recogniser was requested; on live room audio it "
        "cannot hear «компьютер» (1 hypothesis in 25 s of speech)"
    )


def test_feeds_only_the_new_audio(matcher):
    """AcceptWaveform appends to the decoder's context. Re-feeding an
    accumulated buffer made the live recogniser hear the same window repeatedly
    and its hypothesis collapse to ''. Only the new bytes may be handed over."""
    with fake_recognizer(("да",)):
        matcher.begin()
        for _ in range(5):
            matcher.feed(CHUNK)

    rec_sizes = []

    class _SizeSpy(_FakeRec):
        def AcceptWaveform(self, data):
            rec_sizes.append(len(data))
            return super().AcceptWaveform(data)

    with fake_recognizer(("да",)):
        import vosk
        orig = vosk.KaldiRecognizer
        vosk.KaldiRecognizer = _SizeSpy
        try:
            matcher.begin()
            matcher.max_secs = 3600.0   # keep the window open for 5 chunks
            for _ in range(5):
                matcher.feed(CHUNK)
        finally:
            vosk.KaldiRecognizer = orig

    assert rec_sizes == [len(CHUNK)] * 5, (
        f"recogniser got {rec_sizes} — each call must carry only new audio"
    )


def test_fires_exactly_once_per_window(matcher):
    """The session calls _fire_wake() on every True, so a second True would
    restart the wake."""
    with fake_recognizer(("компьютер", "компьютер", "компьютер")):
        matcher.begin()
        results = [matcher.feed(CHUNK) for _ in range(4)]
    assert results[0] is True
    assert results.count(True) == 1
    assert matcher.triggers == 1
    assert matcher.last_text == "компьютер"


def test_no_wake_while_nothing_matches(matcher):
    with fake_recognizer(("", "да", "включи", "")):
        matcher.begin()
        assert [matcher.feed(CHUNK) for _ in range(4)] == [False] * 4
    assert matcher.triggers == 0
    assert matcher.last_partial == ""


def test_max_secs_bounds_by_audio_not_wall_clock(matcher):
    """The context bound must count AUDIO RECEIVED.

    With a wall-clock bound the recogniser was discarded every 3 s having
    consumed almost nothing (the wake path used to be VAD-gated, so only ~5 % of
    chunks reached it), and its hypothesis never developed — 1 hit in 5 live
    attempts against 4 decodes of the same recording when fed continuously.
    """
    m = matcher
    with fake_recognizer(("да", "да", "да")):
        m.begin()
        m.max_secs = 999.0     # huge in seconds ...
        m._t0 -= 3600.0        # ... and an hour of wall clock later
        assert m.feed(CHUNK) is False
    assert m.active is True, "wall clock must NOT end the window"

    built = []

    class _CountRec(_FakeRec):
        def __init__(self, model, sr, grammar=None):
            built.append(1)
            super().__init__(model, sr, grammar)

    # the window must ROLL OVER, not go dead: a bare reset() left _rec = None
    # and every later feed() returned False until the caller called begin()
    with fake_recognizer(("да", "да")):
        import vosk
        orig = vosk.KaldiRecognizer
        vosk.KaldiRecognizer = _CountRec
        try:
            m.max_secs = 0.1       # 0.1 s of AUDIO
            m.begin()
            for _ in range(6):
                m.feed(CHUNK)      # 6 x 0.2 s
        finally:
            vosk.KaldiRecognizer = orig

    assert m.active is True, "the window must roll over, not go dead"
    assert len(built) > 1, "the recogniser must actually be rebuilt"
    assert m.triggers == 0


def test_default_window_is_long_enough_to_hold_the_word():
    """Regression guard for the measured miss rate.

    On an 84 s livingroom recording with five spoken wake words, a 3 s context
    window detected 2 and an 8 s window detected 4 — the reset was cutting
    through the word itself. 8 s and 30 s score the same, so the default sits
    above both.
    """
    assert vosk_wake.VoskWakeMatcher.__init__.__defaults__[0] >= 8.0


def test_flush_closes_the_window_without_firing(matcher):
    """flush() has nothing to decide — the recogniser reports as it goes — but it
    must still hand a clean window to the next utterance."""
    with fake_recognizer(("да",)):
        matcher.begin()
        assert matcher.flush() is False
    assert matcher.active is False


def test_flush_when_idle(matcher):
    assert matcher.flush() is False


def test_reset_keeps_lifetime_counters(matcher):
    """`triggers` is the "why did it not fire?" evidence in the logs; zeroing it
    on every reset erased exactly the information needed to debug a mute room."""
    matcher.triggers = 5
    matcher.decodes = 3
    matcher.reset()
    assert matcher.triggers == 5
    assert matcher.decodes == 3
    assert matcher.active is False

# --- a second speaker's «компьютер» left no trace at all (06.10.2026) --------
#
# Measured: a second voice said the wake word in the living room and the room
# ignored her for twenty minutes. The log read `suppressed=0` and
# `drop[muted=395 echo=69]` — the gate never refused and the mic never muted — so
# the decoder simply never fired, and BOTH remaining explanations are silent:
# it did not decode the word, or it decoded a spelling outside WAKE_TOKENS and
# transcript_has_wake dropped it. Five spellings are allowed because five were
# observed, so a different voice lands outside the list by construction.


def test_a_near_miss_is_reported_and_never_matched():
    """The distinction that needs different fixes: heard-and-discarded by the
    allowlist, versus not heard at all."""
    from vosk_wake import wake_near_miss, transcript_has_wake, WAKE_TOKENS

    # Outside the list but unmistakably the word. NOT «компьытер» — that one is
    # already allowed, which is exactly how the list got its five entries.
    for spelling in ("компъюта", "компьютэ", "компютера", "компьытеро"):
        assert transcript_has_wake(spelling) is False, (
            f"{spelling!r} must not match — the allowlist is exact on purpose"
        )
    assert wake_near_miss("компъюта") == "компъюта"
    assert wake_near_miss("компьютэ") == "компьютэ"
    assert wake_near_miss("датте включи свет") == "", (
        "ordinary speech must not raise the near-miss alarm"
    )
    # Every allowed spelling stays allowed, and never reports itself.
    for tok in WAKE_TOKENS:
        assert transcript_has_wake(tok) is True
        assert wake_near_miss(tok) == ""


def test_the_near_miss_counter_survives_until_it_is_read():
    """It has to be visible in `vosk diag` as well as in the log, because the log
    is rare and the diag is what gets read after a silent room."""
    import json
    import numpy as np

    # Built the way tests/test_vosk_wake.py builds one — via __new__, so no 88 MB
    # model is loaded — which is also the case the class-level defaults exist for.
    import vosk_wake

    m = vosk_wake.VoskWakeMatcher.__new__(vosk_wake.VoskWakeMatcher)
    m.model_path = "fake"
    m.max_secs = 1.0
    m._model = object()
    m.triggers = 0
    m.decodes = 0
    m.near_misses = 0
    m.last_near_miss = ""
    m.reset()
    # No begin(): it would call vosk.KaldiRecognizer against the fake model. The
    # recogniser is stubbed below, which is the whole point of this test.
    # Drive it with a hypothesis that is close but not allowed: the thing under
    # test is the counter and the report, not the model.
    class _Rec:
        def __init__(self):
            self.n = 0

        def AcceptWaveform(self, pcm):
            self.n += 1
            return self.n > 1

        def Result(self):
            return json.dumps({"text": "компъюта"})

        def PartialResult(self):
            return json.dumps({"partial": ""})

    m._rec = _Rec()
    chunk = np.zeros(2560, dtype=np.int16).tobytes()
    assert m.feed(chunk) is False, "a near miss must not fire the wake"
    assert m.feed(chunk) is False
    assert m.near_misses >= 1
    assert m.last_near_miss == "компъюта"
