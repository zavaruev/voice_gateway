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
import logging
import os
import re
import threading
import time
from array import array
from collections import deque

SR = 16000
# A recognised address word in a NEW final segment is a NEW attempt, even if
# the hypothesis still holds the previous one's token. Must match the camera's
# same-breath debounce so a word's tail cannot fire twice.
_WAKE_REARM_S = 3.0
# A late FINAL may re-arm a wake only if FRESH loud audio arrived after the
# debounce — the same word's tail finalised late carries no new loud chunk.
# 3000 is the camera's measured wake gate floor (CAMERA_WAKE_MIN_PEAK), so a
# chunk counted here is one that could itself have produced a wake.
_WAKE_REARM_PEAK = 3000

# Token forms vosk actually produced for «компьютер» on this room's audio, plus
# the plausible misspellings. An ALLOWLIST, not a prefix match: "компот" starts
# with "комп" and woke the matcher in a test, which is exactly the kind of false
# activation this whole exercise exists to remove.
_log = logging.getLogger("vosk_wake")

WAKE_TOKENS = frozenset({
    "компьютер", "комп", "компютер", "компъютер", "компьытер",
})

# The second address is «домовой» (26.11.2026). The previous address «добби»
# was a rare proper name whose mis-hearings needed an endless allowlist. The
# clean-synthesis screen for «домовой» was MISLEADING: on the user's live
# voice in the living room the small model writes «ты ногой включи свет»,
# «давай включи свет» and «два выключи свет» — never the word itself.
# Measured over 1049 s of the LIVE streaming dump (gated_probe.py): the three
# real attempts fire and television does not. The big model reads all three
# but is ruled out (hardware cannot sustain it), so these forms are accepted
# ONLY at the START of the hypothesis and IMMEDIATELY before a command verb.
# «два раза», «давай вперед» etc. stay ordinary speech. These are MATCH tokens
# only — they must never enter the strip allowlist, where they would corrupt
# unrelated sentences.
DOMOVOY_TOKENS = frozenset({"домовой"})
DOMOVOY_GATED_FORMS = frozenset({"ты ногой", "давай", "два"})
_DOMOVOY_COMMAND_WORDS = frozenset({
    "включи", "выключи", "включить", "выключить", "поставь", "открой",
    "закрой", "сделай", "переключи", "выключите", "включите",
    "погаси", "запусти", "останови", "обнули", "сними",
    "какая", "какие", "сколько", "когда", "где", "как", "что",
    "почему", "зачем", "погода", "времени",
})


def _domovoy_hits(text: str, tokens=frozenset()) -> int:
    """Count only the LIVE gated forms; exact tokens are counted by the caller."""
    t = (text or "").lower().replace("ё", "е").strip(" .,!?;:")
    words = [w.strip(".,!?;:") for w in t.split() if w.strip(".,!?;:")]
    hits = 0
    for form in DOMOVOY_GATED_FORMS:
        n = len(form.split())
        if words[:n] != form.split():
            continue
        rest = words[n:]
        if rest and rest[0] in _DOMOVOY_COMMAND_WORDS:
            hits += 1
        break
    return hits

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


def wake_near_miss(text: str, tokens=WAKE_TOKENS) -> str:
    """The token this text almost was, or "".

    `tokens` defaults to the computer allowlist for backward compatibility; the
    session passes the union of both words when a second wake word is enabled,
    so an inflection of either one is diagnosed instead of silently dropped.

    **The single most useful line in this file, and it did not exist.** Measured
    06.10.2026: a second speaker said «компьютер» in the living room and the room
    ignored her for twenty minutes. The log said `suppressed=0` and
    `drop[muted=395 echo=69]` — so the gate never refused, the mic never muted,
    and the decoder simply never fired. **Both remaining explanations are silent**:
    either it did not decode the word, or it decoded a spelling outside
    `WAKE_TOKENS` and `transcript_has_wake` dropped it on the floor. Five
    spellings are allowed because five were OBSERVED, which means a different
    voice lands outside the list by construction.

    The check is deliberately cheap: a shared STEM prefix rejects almost every
    hypothesis in the room, and only survivors are compared. It was 3 characters
    until «домовой» arrived (26.11.2026): the «дом» stem makes «доме», «домой»,
    «домик» and «домашний» look like near misses on television, which floods
    `near=` and hides a real missed spelling. 5 characters still catches every
    inflection of both addresses («компъюта», «домовому»). It is a diagnostic,
    never a match — `transcript_has_wake` stays an exact test, because widening
    it to a prefix admits «компот» and «компания».
    """
    tokens = tokens or WAKE_TOKENS
    t = (text or "").lower().replace("ё", "е")
    if not t.strip():
        return ""
    for tok in t.split():
        bare = tok.strip(".,!?;:")
        if bare in tokens:
            # Already allowed — reporting it would be noise. The first version
            # missed this and every allowed spelling came back as its own near
            # miss, because «комп» is a prefix of every other entry in the list.
            continue
        for cand in tokens:
            if cand.startswith(bare) or bare.startswith(cand):
                # A prefix is a truncation, not a misspelling: «комп» and
                # «компю» are on the allowlist's own reasoning, not near misses.
                continue
            n = min(len(bare), len(cand), 5)
            if n and bare[:n] == cand[:n] and abs(len(bare) - len(cand)) <= 3:
                return bare
    return ""


def transcript_has_wake(text: str, tokens=WAKE_TOKENS) -> bool:
    """Exact-token match against the observed spellings of the wake word.

    Tolerant only where it was measured to be necessary: vosk drops letters on
    short/noisy audio ("комп", "компютер"), so those forms are listed. Widening
    to a "комп..." prefix test admits «компот» and «компания», which is a false
    activation per occurrence.
    """
    t = (text or "").lower().replace("ё", "е")
    # Check every token: vosk may emit filler words («ну», «это») before the
    # wake word on short/noisy audio, so the addressee can land anywhere in the
    # phrase. The match is exact against observed spellings — «компот» and
    # «компания» are NOT in WAKE_TOKENS and stay rejected.
    words = t.split()
    if not words:
        return False
    if "домовой" in tokens and _domovoy_hits(t, tokens):
        return True
    return any(tok.strip(".,!?;:") in tokens for tok in words)


def wake_occurrences(text: str, tokens=WAKE_TOKENS) -> int:
    """How many times the wake word appears in this hypothesis.

    The recogniser's partial is a GROWING transcript of the current segment, so
    this is what tells the SAME word from a NEW one: the tail of the word we
    just heard leaves the count at one, and a word the user says again raises
    it to two.
    """
    t = (text or "").lower().replace("ё", "е")
    count = sum(
        1 for tok in t.split()
        if tok.strip(".,!?;:") in tokens
    )
    if "домовой" in tokens:
        count += _domovoy_hits(t, tokens)
    return count


def wake_kind_in(text: str, alt_tokens=frozenset(), alt_word: str = "") -> str:
    """Which address word is CLOSEST TO THE COMMAND in this hypothesis.

    The recogniser's partial is a growing transcript, so when one breath holds
    both words the one nearer the end is the one the user is addressing now.
    Unknown text returns "" — callers only use this after a wake was confirmed.
    """
    t = (text or "").lower().replace("ё", "е")
    # The address closest to the command wins when one breath holds both.
    words = [w.strip(".,!?;:") for w in t.split()]
    alt_at = [i for i, w in enumerate(words) if w in alt_tokens]
    comp_at = [i for i, w in enumerate(words) if w in WAKE_TOKENS]
    gated = "домовой" in alt_tokens and _domovoy_hits(t, alt_tokens)
    if not alt_at and not gated:
        return "computer" if comp_at else ""
    alt_pos = max(alt_at) if alt_at else 0
    if not comp_at or alt_pos > comp_at[-1]:
        return alt_word or "dobbi"
    return "computer"


class VoskWakeMatcher:
    # Class-level defaults, because tests and any future caller may build an
    # instance with `__new__` and skip `__init__` — `tests/test_vosk_wake.py`
    # does exactly that to avoid loading an 88 MB model. A missing attribute on
    # this counter would raise inside `feed()`, i.e. inside the live audio path,
    # which is the worst place to discover a test-only shortcut.
    near_misses = 0
    last_near_miss = ""
    # How many times the wake word is already in the recogniser's hypothesis.
    # A class-level default, for the same reason as the rest of them: a matcher
    # built with `__new__` must not raise inside `feed()`.
    _wake_hits = 0
    # Second, opt-in address word. Defaults make a matcher built with __new__
    # (tests, and any future caller) behave like the old computer-only path.
    alt_word = ""
    alt_tokens = frozenset()
    last_kind = "computer"
    # A wake token present in the hypothesis but NOT a new occurrence used to
    # be the most expensive silence in the path: no fire and no log line at all.
    # Lifetime counter, reported by `vosk diag`, like `triggers`.
    blocked_same = 0
    _last_fire_wall = 0.0
    # One recogniser segment may contain the tail of several words. A DIFFERENT
    # address word appended inside the same segment is not a new wake: firing on
    # it would call _fire_wake twice and destroy the command being collected.
    _segment_fired = False
    # A short address («добби») is often finalised ~2 s AFTER the loud audio
    # that produced it: the trailing 1.9 s window is already quiet and the
    # camera's energy gate called a real wake "silence". Keep the peaks of the
    # whole recogniser segment (bounded to ~8 s) so the gate can judge the
    # audio that produced the word instead of the silence at its finalisation.
    # Class-level defaults so a matcher built with `__new__` (tests) still has
    # them — a missing attribute here raises inside the live audio path.
    _peak_win: deque = deque(maxlen=50)
    last_from_final = False

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

    def __init__(self, model_path: str, max_secs: float = 30.0,
                 alt_tokens=frozenset(), alt_word: str = ""):
        self.alt_tokens = frozenset(alt_tokens)
        self.alt_word = alt_word or ("dobbi" if self.alt_tokens else "")
        self.last_kind = "computer"
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
        self._peak_win = deque(maxlen=50)
        # Parallel segment tags for the peak window: a FINAL must be judged on
        # the audio of ITS segment, not on a loud burst from up to 8 s earlier.
        self._peak_seg = deque(maxlen=50)
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
        self._segment_fired = False
        self._wake_hits = 0
        # Fresh loud chunks fed AFTER the last fire + debounce. The evidence
        # that a late FINAL is a NEW word rather than the old word's tail.
        self._fresh_after_fire = 0
        self._fed = 0
        self._t0 = 0.0
        self.last_text = ""
        self.last_trigger_text = ""
        self.last_partial = ""
        self._peak_win.clear()
        # Hand-built sessions (VoskWakeMatcher.__new__ in the tests) have no
        # instance deques yet — the same trap as other new attributes here.
        if getattr(self, "_peak_seg", None) is None:
            self._peak_seg = deque(maxlen=50)
        self._peak_seg.clear()
        self._seg = 0
        self._final_seg = None
        self.last_from_final = False

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
        self._segment_fired = False
        self._wake_hits = 0
        self._fresh_after_fire = 0
        self._last_fire_wall = 0.0
        self._fed = 0
        self._t0 = time.time()
        # A lifetime counter like `triggers`, deliberately NOT cleared by reset().
        # Reported because it is otherwise the most expensive silence in this
        # file: the room ignores a person and every counter reads clean.
        self.near_misses = 0
        self.last_near_miss = ""
        self._peak_win.clear()
        # Hand-built sessions (VoskWakeMatcher.__new__ in the tests) have no
        # instance deques yet — the same trap as other new attributes here.
        if getattr(self, "_peak_seg", None) is None:
            self._peak_seg = deque(maxlen=50)
        self._peak_seg.clear()
        self._seg = 0
        self._final_seg = None
        self.last_from_final = False

    def _segment_peaks(self) -> list:
        """Peaks belonging to the segment that produced the last FINAL.

        A hand-built test session may set only `_peak_win` (the older shape);
        with no segment tags, fall back to the whole window rather than dying
        inside the live audio path.
        """
        segs = getattr(self, "_peak_seg", None)
        final_seg = getattr(self, "_final_seg", None)
        if not segs or len(segs) != len(self._peak_win) or final_seg is None:
            return list(self._peak_win)
        return [p for p, s in zip(self._peak_win, segs) if s == final_seg]

    def wake_evidence_peak(self) -> int:
        """Peak of the audio that produced a FINAL-segment wake, else 0.

        A PARTIAL hypothesis is still growing, so its trailing window is the
        only honest evidence. A FINAL segment is a closed utterance: the word
        may have been spoken seconds before the segment arrived, and that is
        exactly when the camera's 1.9 s window has gone quiet (measured
        22.11.2026, «добби включи свет» finalised 1.92 s after a 22 887 peak).
        """
        if not self.last_from_final:
            return 0
        peaks = self._segment_peaks()
        return int(max(peaks)) if peaks else 0

    def wake_evidence_loud(self, bar: int) -> int:
        """How many chunks OF THE FINAL'S SEGMENT reached `bar` — the click guard."""
        if not self.last_from_final:
            return 0
        return sum(1 for p in self._segment_peaks() if p >= bar)

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

        self._fed += len(pcm)
        # Keep the segment's own energy. The camera gate used to judge a FINAL
        # wake on its trailing 1.9 s window, which is quiet by the time a short
        # word is finalised — a real «добби» was refused as "silence".
        peak = max((abs(v) for v in array("h", pcm)), default=0)
        self._peak_win.append(peak)
        self._peak_seg.append(self._seg)
        # Evidence that a LATE final belongs to a NEW word, not to the tail of
        # the one that already fired: loud audio that arrived after the
        # same-breath debounce. A word finalised late has all its loud chunks
        # BEFORE the fire, so this stays 0 and the final cannot re-arm.
        if (self._last_fire_wall and peak >= _WAKE_REARM_PEAK
                and time.time() - self._last_fire_wall > _WAKE_REARM_S):
            self._fresh_after_fire += 1
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
        tokens = WAKE_TOKENS | self.alt_tokens
        if self._rec.AcceptWaveform(pcm):
            self.last_from_final = True
            text = json.loads(self._rec.Result()).get("text", "")
            # This FINAL closes the current segment. Record its index BEFORE
            # advancing: the chunk just appended (and the ~2 s of pre-final
            # audio that produced the word) belong to it, while anything fed
            # from now on belongs to the next segment and must not lend its
            # peaks to this final's click guard.
            self._final_seg = self._seg
            self._seg += 1
            # A closed recogniser segment is an utterance boundary: after it a
            # different address word is a NEW wake, not the tail of the last one.
            self._segment_fired = False
            # Measured 24.11.2026 16:10: `blocked=30` — an address word in the
            # final segment was silently discarded because the OLD hypothesis
            # still held the same token. Re-arm after the same-breath debounce,
            # so a later attempt can fire while the word's tail still cannot.
            if (
                self._last_fire_wall
                and time.time() - self._last_fire_wall > _WAKE_REARM_S
                and transcript_has_wake(text, tokens)
            ):
                # The final's own text still holds the word. Zeroing the count
                # unconditionally made `hits(1) > _wake_hits(0)` fire a SECOND
                # time for one spoken word; camera._fire_wake() then wiped the
                # command being collected. Re-arm only on real evidence of a
                # NEW word: more occurrences than the word that fired, or fresh
                # loud audio after the debounce (`blocked=30` repeat).
                if (wake_occurrences(text, tokens) > self._wake_hits
                        or self._fresh_after_fire):
                    self._wake_hits = 0
                    _log.info("vosk: wake re-armed on a new final segment")
                else:
                    _log.info(
                        "vosk: late final of the same word — not a new wake"
                    )
        else:
            self.last_from_final = False
            text = json.loads(self._rec.PartialResult()).get("partial", "")
        # kept for the periodic diagnostic in camera_client: when the live room
        # stays silent while every probe passes, the decoder's own hypothesis is
        # the only evidence of what it actually hears
        self.last_partial = text
        has_wake = transcript_has_wake(text, tokens)
        if not has_wake:
            near = wake_near_miss(text, tokens)
            if near:
                # A near miss means the model still heard the wake word but in
                # another form (an inflection the allowlist does not list, e.g.
                # «компьютера»). Keep _wake_hits so the same word does not refire
                # on the tail of the same breath; when the user says it AGAIN
                # later, the occurrence count grows and it fires — that is what
                # a wake word is for. Measured 21.10.2026 12:27: seven near-misses
                # of «компьютера» in a row zeroed the count, and the next
                # «компьютер» fired again. Do not reset on a near miss.
                self.near_misses += 1
                self.last_near_miss = near
                _log.warning(
                    "vosk: wake NEAR MISS #%d — decoded %r, not in the wake tokens %s",
                    self.near_misses, near, sorted(tokens),
                )
                return False
            # The word truly left the hypothesis: the next one is a NEW word and
            # may fire again.
            self._wake_hits = 0
            self._segment_fired = False
            return False

        # FIRE ONCE PER SPOKEN WORD.
        #
        # The recogniser's hypothesis is a growing transcript of the CURRENT
        # segment, so the number of times the wake word appears in it is what
        # tells the SAME word from a NEW one:
        #
        #   * the tail of the word we just heard arrives late on a slow link
        #     (the corridor delivered 08:38's word over 2.6 s of wall time),
        #     and the OLD code reset the recogniser right after the fire — so the
        #     fresh recogniser decoded that same tail as a NEW «компьютер» and
        #     fired again. Measured 21.10.2026 08:38-08:39: FIVE fires for one
        #     word, and every fire clears `_vad_speech_buf`, so the command the
        #     user was saying was destroyed five times over and the room
        #     answered «Wake with no command — sending greeting».
        #   * a word the user says AGAIN grows the hypothesis to two
        #     occurrences, and that is a new word and must fire — a wake word
        #     exists so that saying it gets attention, always.
        #
        # So the rule is the OCCURRENCE COUNT, and the recogniser is NOT reset
        # after a fire: resetting it is what made the same word fire again.
        kind = wake_kind_in(text, self.alt_tokens, self.alt_word) or "computer"
        hits = wake_occurrences(text, tokens)
        if hits <= self._wake_hits:
            self.blocked_same += 1
            return False
        if kind != self.last_kind and self._segment_fired:
            # The decoder's hypothesis grows: a different word appended to the
            # segment where we just fired is the same breath's tail. Firing again
            # would call _fire_wake twice and clear the command being collected.
            return False
        self._wake_hits = hits
        # This fire is now the reference; fresh audio before it must not count
        # as evidence for a later re-arm.
        self._fresh_after_fire = 0
        self.last_kind = kind
        self._segment_fired = True
        self._last_fire_wall = time.time()

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
