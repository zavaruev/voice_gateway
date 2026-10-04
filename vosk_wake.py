"""Wake word by DECODING the word, not by scoring the sound.

Why this exists
    Three openWakeWord heads trained on this room all learned envelopes instead
    of the word (see AGENTS.md), and the honest limit is the 27 usable
    recordings, not the classifier. A decoder has no such ceiling: it produces
    text, so "did it hear the wake word" becomes a lookup instead of a
    discrimination problem — and a false activation stops being a probability
    question.

Measured on the livingroom material (36 Whisper-confirmed «компьютер» bursts,
42 speech bursts without it, 420 s of television of which 126 s unseen by any
measurement):

    shipped (free decoder)  5/5 live   3/3 @9 dB SNR   0 false / 7 min of TV
    grammar-constrained     — rejected, see below

A grammar-constrained recognizer (wake word plus 33 filler words) was tried as a
cheap first stage with a free decoder as confirmation. It is 5-6x faster per
window and scored better on isolated 1.3 s bursts, but on LIVE room audio it
produced **one** hypothesis across 25 s of loud speech where the free decoder
produced 27 on the same bytes: restricting the vocabulary makes it reject the
very word the wake word is made of when the speaker is across the room and the
television is on. That whole path was removed. Do not reintroduce it — there is
a test asserting no grammar is passed (`test_matcher_uses_a_free_decoder`).

FEED RAW AUDIO, EVERY CHUNK
    openWakeWord needs per-chunk peak normalisation; a decoder does not and is
    actively hurt by it — every independent 200 ms gain jump is distortion.
    Measured on the same 36 bursts: 94 % recall on raw audio, 44 % on the
    AGC-normalised stream. And do NOT gate the feed on the Silero VAD: it passed
    21 of 419 chunks on a recording where the word was plainly audible, and the
    decoder then saw a sparse stream it could not decode.

CPU (per room, measured)
    AcceptWaveform+PartialResult on 200 ms chunks is ~1.8 ms/chunk ≈ 0.9 % of a
    core, running continuously while the room has audio. `max_secs` (default
    30 s) bounds the recogniser context and must be counted in AUDIO FED, not in
    wall clock — a 3 s window cuts through the wake word itself.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

SR = 16000

# Token forms vosk actually produced for «компьютер» on this room's audio, plus
# the plausible misspellings. An ALLOWLIST, not a prefix match: "компот" starts
# with "комп" and woke the matcher in a test, which is exactly the kind of false
# activation this whole exercise exists to remove.
WAKE_TOKENS = frozenset({
    "компьютер", "комп", "компютер", "компъютер", "компьытер",
})

_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, object] = {}


def get_model(path: str):
    """Load (and cache) the vosk model.

    Eight rooms must share ONE model: KaldiRecognizer instances are per stream
    and cheap, but the model itself is ~88 MB and would be loaded eight times
    (and re-read from disk eight times) otherwise.
    """
    with _MODEL_LOCK:
        m = _MODELS.get(path)
        if m is None:
            import vosk
            vosk.SetLogLevel(-1)
            t0 = time.time()
            m = vosk.Model(path)
            _MODELS[path] = m
            print(f"[vosk] модель {path} загружена за {time.time() - t0:.1f}s")
        return m


def transcript_has_wake(text: str) -> bool:
    """Exact-token match against the observed spellings of the wake word.

    Tolerant only where it was measured to be necessary: vosk drops letters on
    short/noisy audio ("комп", "компютер"), so those forms are listed. Widening
    to a "комп..." prefix test admits «компот» and «компания», which is a false
    activation per occurrence.
    """
    t = (text or "").lower().replace("ё", "е")
    # strip punctuation: vosk usually emits bare words, but a stray '.' or ','
    # in a partial hypothesis must not turn a correct hit into a miss
    return any(tok.strip(".,!?;:") in WAKE_TOKENS for tok in t.split())


class VoskWakeMatcher:
    """Per-stream wake-word matcher built on a FREE (unconstrained) decoder.

    NOT thread-safe by itself. One instance per camera session; feed raw 16 kHz
    mono int16 with `feed()` and it returns True once, on the chunk where the
    decoder produced the wake word.

    WHY FREE AND NOT GRAMMAR-CONSTRAINED
        A two-stage design (fast grammar trigger + free confirmation) was tried
        and removed. The grammar recogniser was 5-6x cheaper, but on the LIVE
        room audio it produced **one** hypothesis across 25 s of loud speech
        while the free decoder produced 27 on the same bytes. Restricting the
        decoder to «компьютер» plus 33 fillers rejects exactly the word the wake
        word is made of when the speaker is across the room and the television
        is on — it had 100 % recall on isolated 1.3 s bursts and 0 % on live
        audio. Speed is not worth a detector that cannot hear its own word:
        with the free decoder as the trigger, planted words were detected 3/3 at
        9 dB SNR in live room audio.
    """

    def __init__(self, model_path: str, max_secs: float = 30.0):
        self.model_path = model_path
        # Context bound only, measured in AUDIO FED. The room's VAD never reports
        # silence, so something has to stop the recogniser accumulating a whole
        # session — but keep the window long: at 3 s it cut through the wake
        # word itself and lost 2 of 4 detections on an 84 s recording where the
        # word was plainly audible (8 s and above recover all of them, and vosk
        # segments on its own at pauses so a long window costs nothing).
        self.max_secs = max_secs
        self._model = get_model(model_path)
        # lifetime counters, intentionally NOT cleared by reset()
        self.triggers = 0
        self.decodes = 0
        self.reset()

    def reset(self):
        """Clear any in-flight utterance. Safe to call at any time.

        Deliberately does NOT touch `triggers`/`decodes`: those are lifetime
        counters for the "why did it not fire?" question in the logs, and
        zeroing them here made every abandoned utterance erase the evidence
        that it had even proposed a wake word.
        """
        self._rec = None
        self._fired = False
        self._fed = 0
        self._t0 = 0.0
        self.last_text = ""
        self.last_trigger_text = ""
        self.last_partial = ""

    def begin(self):
        """(Re)start the recogniser for a fresh window.

        Called by the owner when no window is open, and by `feed()` itself when
        `max_secs` of audio has been consumed. It is not tied to VAD onsets: the
        room's Silero VAD never reports silence (0 `speech=False` in 15 live
        minutes), so there is no utterance boundary to hang this on.
        """
        import vosk
        self._rec = vosk.KaldiRecognizer(self._model, SR)
        self._rec.SetWords(True)
        self._fired = False
        self._fed = 0
        self._t0 = time.time()

    @property
    def active(self) -> bool:
        return self._rec is not None

    def feed(self, pcm: bytes) -> bool:
        """Consume one chunk of RAW int16 16 kHz. True on the chunk where the
        decoder produced the wake word.

        Returns False while idle or when nothing matched.
        """
        if self._rec is None:
            return False
        if self._fired:
            return False

        self._fed += len(pcm)
        # Bound the context by AUDIO RECEIVED, not by wall clock. With a wall
        # clock bound and a sparse feed the recogniser was thrown away every
        # 3 seconds having consumed almost nothing, so its hypothesis never
        # developed. Count samples.
        #
        # `begin()` rather than `reset()`: a bare reset leaves `_rec = None` and
        # every later feed() returns False until the caller happens to call
        # begin() again. Self-healing here removes that ordering dependency
        # entirely, and the chunk that crossed the bound is still analysed.
        if self._fed >= int(self.max_secs * SR) * 2:
            self.reset()
            self.begin()
            self._fed = len(pcm)

        # Feed ONLY the new audio. AcceptWaveform appends to the decoder's own
        # context, so re-feeding an accumulated buffer makes the recogniser hear
        # the same window again on every chunk and its hypothesis collapses to ''
        # — which is what the live room showed (triggers=0, hyp='') with loud
        # speech present, while every offline probe that fed each chunk once
        # passed.
        if self._rec.AcceptWaveform(pcm):
            text = json.loads(self._rec.Result()).get("text", "")
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")
        # kept for the periodic diagnostic in camera_client: when the live room
        # stays silent while every probe passes, the decoder's own hypothesis is
        # the only evidence of what it actually hears
        self.last_partial = text
        if not transcript_has_wake(text):
            return False

        self.triggers += 1
        self.last_trigger_text = text
        self.last_text = text
        self.decodes += 1
        self._fired = True
        return True

    def flush(self) -> bool:
        """Utterance ended: close the window so the next one starts clean.

        Always False — the recogniser reports as it goes, so there is nothing
        left to decide. Kept because `_vad_process` calls it on silence and
        because a fresh window is the correct behaviour there.
        """
        if self._rec is None:
            return False
        self.reset()
        return False
