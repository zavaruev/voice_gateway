"""Two wake words in one decoder: Компьютер (normal) and Добби (pilot mode).

The tokens for Добби come from a live recording; an empty allowlist is the
default, in which case this module must behave exactly as before (the old
allowlist, the old pip/ack behaviour).

NOTE: imports `camera_client` (and thus `av`), so this module runs inside the
image, like tests/test_wake_gates.py.
"""
import asyncio
import json
from collections import deque
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import vosk_wake
from camera_client import CameraConfig, CameraSession


CHUNK = np.zeros(2560, dtype=np.int16).tobytes()
LOUD_CHUNK = np.full(2560, 20000, dtype=np.int16).tobytes()
DOMOVOY = ("домовой",)


class _ScriptedRec:
    def __init__(self, partial=""):
        self.partial = partial

    def AcceptWaveform(self, pcm):
        return False

    def Result(self):
        return json.dumps({"text": self.partial})

    def PartialResult(self):
        return json.dumps({"partial": self.partial})


def _matcher(alt=DOMOVOY):
    m = vosk_wake.VoskWakeMatcher.__new__(vosk_wake.VoskWakeMatcher)
    m.model_path = "fake"
    m.max_secs = 1.0
    m._model = object()
    m.triggers = 0
    m.decodes = 0
    m.alt_word = "dobbi" if alt else ""
    m.alt_tokens = frozenset(alt)
    m.reset()
    m._rec = _ScriptedRec()
    return m


# --- recogniser -----------------------------------------------------------

def test_second_word_fires_and_is_reported_as_its_own_kind():
    m = _matcher()
    m._rec.partial = "домовой"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"

    m.reset()
    m._rec = _ScriptedRec("компьютер")
    assert m.feed(CHUNK) is True
    assert m.last_kind == "computer"


def test_a_simultaneous_hypothesis_picks_the_closest_address():
    """A FINAL segment delivered whole (both words already in it) picks the
    address nearest the command. The NORMAL incremental path is different: the
    first word fires on its own chunk and the later one is blocked by the
    same-breath guard — that case is tested immediately below."""
    m = _matcher()
    m._rec.partial = "компьютер домовой"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi", "the address closest to the command wins"
    assert m.feed(CHUNK) is False, "the tail of one breath must not fire again"
    assert m.triggers == 1


def test_a_second_word_appended_after_a_fire_is_not_a_new_wake():
    """The hypothesis GROWS. «компьютер» then «компьютер добби» must not fire
    twice — the second fire would cancel the command being collected."""
    m = _matcher()
    m._rec.partial = "компьютер"
    assert m.feed(CHUNK) is True
    m._rec.partial = "компьютер домовой"
    assert m.feed(CHUNK) is False
    # The words leave the hypothesis: the next address is a new one.
    m._rec.partial = ""
    assert m.feed(CHUNK) is False
    m._rec.partial = "домовой"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"


def test_dobbi_does_not_match_when_its_allowlist_is_empty():
    m = _matcher(alt=())
    m._rec.partial = "домовой"
    assert m.feed(CHUNK) is False, "empty alt allowlist must leave the old path alone"
    m._rec.partial = "компьютер"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "computer"


def test_a_misspelling_of_the_second_word_is_a_diagnostic_not_a_match():
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOMOVOY)
    assert vosk_wake.transcript_has_wake("домовому", tokens) is False
    assert vosk_wake.wake_near_miss("домовому", tokens) == "домовому"


def test_near_miss_ignores_ordinary_home_words():
    """«дом» is a frequent Russian stem. Reporting «доме/домой/домашний» as
    near misses would flood `near=` on television and make a real missed
    spelling of «домовой» undiagnosable. Only inflections of the full stem
    count — the same rule that keeps «компъюта» visible for «компьютер»."""
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOMOVOY)
    for ordinary in ("доме", "дома", "домой", "домик", "домашний"):
        assert vosk_wake.wake_near_miss(ordinary, tokens) == "", ordinary
    assert vosk_wake.wake_near_miss("домовому", tokens) == "домовому"


def test_wake_kind_in_uses_the_last_word_in_the_hypothesis():
    fn = vosk_wake.wake_kind_in
    assert fn("компьютер домовой", frozenset(DOMOVOY)) == "dobbi"
    assert fn("домовой компьютер", frozenset(DOMOVOY)) == "computer"
    assert fn("включи свет", frozenset(DOMOVOY)) == ""


# --- per-word acknowledgement behaviour -----------------------------------

def _wake_session():
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "livingroom"
    s._wake_kind = "computer"
    s._attention_pip = True
    s._wake_ack_enabled = True   # deliberately the stale pilot values
    s._wake_ack_delay = 2.0
    s._wake_ack_min_gap_s = 20.0
    s._last_wake_ack_at = 0.0
    s._wake_ack_gen = 0
    s._wake_ack_task = None
    s._wake_ack_pool = ["Слушаю."]
    s._wake_ack_idx = 0
    s._wake_timeout = 60.0
    s._awaiting_command = False
    s._ww_base_thresh = 0.3
    s._ww_thresh = 0.3
    s._ww_consec = 0
    s._ww_recent = []
    s._vad_has_speech = True
    s._vad_speech_buf = bytearray(b"\x01\x02" * 100)
    s._preroll = []
    s._vad_speech_consecutive = 0
    s._speech_starts = 0
    s._last_speech_at = 0.0
    s._endpoint = None
    s._vad_silence_frames = 0
    s._stream_cancel = lambda: None
    s._stream_forced_text = ""
    s._recent_peaks = deque([1000], maxlen=12)
    s._noise = type("N", (), {"value": 0.01})()
    return s


@pytest.mark.asyncio
async def test_computer_wake_forces_pip_and_no_spoken_ack():
    """«Компьютер» must behave like every other camera even in the pilot room."""
    s = _wake_session()
    with (
        patch.object(s, "_play_attention", new=AsyncMock()) as pip,
        patch.object(s, "_arm_wake_ack") as ack,
    ):
        await s._fire_wake("vosk", kind="computer")
        await asyncio.sleep(0)
    assert pip.await_count == 1, "the address must still get its beep"
    assert ack.call_count == 0, "no spoken acknowledgement after «компьютер»"
    assert s._wake_kind == "computer"


@pytest.mark.asyncio
async def test_dobbi_wake_forces_spoken_ack_and_no_pip():
    """«Добби» keeps the current living-room pilot: no beep, a short line."""
    s = _wake_session()
    s._attention_pip = True
    s._wake_ack_enabled = False
    with (
        patch.object(s, "_play_attention", new=AsyncMock()) as pip,
        patch.object(s, "_arm_wake_ack") as ack,
    ):
        await s._fire_wake("vosk", kind="dobbi")
    assert pip.await_count == 0, "the pilot mode has no beep"
    assert ack.call_count == 1, "the short phrase must be armed"


def test_dobbi_uses_a_lower_loud_chunk_bar_than_kompjyuter():
    """The energy gate was calibrated on a 4-syllable word (9-10 loud chunks).
    Live 24.11.2026: a decoded «добби» with loud=2/3 was refused as 'silence'
    while a later «добби» with loud=10/12 fired. The short word needs bar 2;
    a click is still only one loud chunk."""
    s = CameraSession.__new__(CameraSession)

    class _M:
        last_kind = "dobbi"

    s._vosk_wake = _M()
    assert s._wake_loud_bar() == 2
    s._vosk_wake.last_kind = "computer"
    assert s._wake_loud_bar() == 3


def test_big_model_loud_bar_override_applies_to_both_words():
    """vosk-model-ru-0.42 commits to «компьютер» at loud=1-2/12, so the
    small-model bars (3/2) would refuse a word it heard correctly. One override
    covers both words; 0 keeps the proven defaults for every other room."""
    s = CameraSession(CameraConfig(stream_name="livingroom", wake_loud_bar=1))

    class _M:
        last_kind = "computer"

    s._vosk_wake = _M()
    assert s._wake_loud_bar() == 1
    s._vosk_wake.last_kind = "dobbi"
    assert s._wake_loud_bar() == 1
    other = CameraSession(CameraConfig(stream_name="kitchen"))
    other._vosk_wake = _M()
    assert other._wake_loud_bar() == 3


@pytest.mark.asyncio
async def test_dobbi_ack_actually_speaks_with_a_stale_config_flag():
    """Patching `_arm_wake_ack` hides this: the task it arms checks
    `_wake_ack_enabled`, which is still false from the old living-room env."""
    s = _wake_session()
    s._wake_ack_enabled = False
    s._wake_ack_delay = 0.0
    s._wake_ack_min_gap_s = 0.0
    s._speech_run = 0
    s._last_speech_at = 0.0

    async def _tts(_text, speed=1.0):
        return b"pcm"

    s._tts_fetch = _tts
    s._speak_pcm = AsyncMock()
    with patch.object(s, "_play_attention", new=AsyncMock()):
        await s._fire_wake("vosk", kind="dobbi")
    assert s._wake_ack_task is not None
    await s._wake_ack_task
    assert s._speak_pcm.await_count == 1, "the pilot address said nothing"


@pytest.mark.asyncio
async def test_double_address_still_gets_a_short_acknowledgement():
    """No reaction at all is the worst outcome: the user repeats the word and
    creates a second fire. A bare address must get a short line, while a
    command in the same breath stays silent."""
    import time
    s = CameraSession(CameraConfig(stream_name="livingroom"))
    s.wake_keyword = "компьютер"
    s._wake_tokens = frozenset({"компьютер", "комп", "домовой"})
    s._wake_kind = "dobbi"
    s._wake_detected = True
    s._wake_only_pending = False
    s._wake_ack_enabled = True
    s._last_wake_fired_at = time.time() - 0.7
    armed = []

    def _arm(delay=None):
        armed.append(delay)

    s._arm_wake_ack = _arm
    await s._handle_wake_or_command("Домовой Домовой", "camera")
    assert armed == [0.8], "a bare address was answered with silence"


@pytest.mark.asyncio
async def test_a_late_wake_keeps_the_command_being_collected_for_both_words():
    s = _wake_session()
    s._awaiting_command = True
    before = bytes(s._vad_speech_buf)
    with (
        patch.object(s, "_play_attention", new=AsyncMock()),
        patch.object(s, "_arm_wake_ack"),
    ):
        await s._fire_wake("vosk", kind="dobbi")
    assert bytes(s._vad_speech_buf) == before, "a second fire destroyed the command"


# --- env plumbing ---------------------------------------------------------

def test_the_old_dobbi_mishearing_family_is_refused():
    """26.11.2026: those forms existed only because «добби» is a rare proper
    name and lost Vosk's LM ranking. With «домовой» the whole gated family is
    gone — a TV phrase containing any of them must not fire a wake."""
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOMOVOY)
    for text in ("добрый включи свет", "дарби выключи свет",
                 "да выключи свет", "доме какая погода на улице",
                 "домика включи свет", "давай без включи свет",
                 "давай при какая погода", "тоби выключи свет",
                 "зомби включи свет", "дабби", "додби", "тоби"):
        assert vosk_wake.transcript_has_wake(text, tokens) is False, text
    for text in ("добрый включи свет", "дарби выключи свет",
                 "доме какая погода на улице", "давай без включи свет"):
        m = _matcher()
        m._rec.partial = text
        assert m.feed(CHUNK) is False, text
        assert m.triggers == 0


def test_live_domovoy_mishearings_fire_only_before_a_command_verb():
    """26.11.2026, living-room Vosk dump: the speaker's «Домовой, включи
    свет» was written by the SMALL model as «ты ногой включи свет» and
    «давай включи свет»; «выключи» arrived as «два выключи свет». Those exact
    forms are the address — accept them only at the START and IMMEDIATELY
    before a command verb. «два раза» on television must stay ordinary."""
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOMOVOY)
    for text in ("ты ногой включи свет", "давай включи свет",
                 "два выключи свет", "домовой включи свет"):
        assert vosk_wake.transcript_has_wake(text, tokens) is True, text
        assert vosk_wake.wake_kind_in(text, frozenset(DOMOVOY), "dobbi") == "dobbi", text
    for text in ("ты ногой", "ты ногой в", "давай", "давай вперед",
                 "два", "два раза", "дорогой включи свет",
                 "ты ногой человек включи свет"):
        assert vosk_wake.transcript_has_wake(text, tokens) is False, text


def test_living_room_pause_endpoint_is_relaxed_for_the_short_address():
    """26.11.2026 replay of 11 recorded living-room windows (pause_replay.py):
    M=5/R=15 ended 2.88-4.00 s after the window; M=3/R=10 ended 0.8 s sooner
    in every window and NEVER before the command verb was in the hypothesis."""
    import os
    compose = "/ai-prod/docker-compose.yml"
    if not os.path.exists(compose):
        pytest.skip("compose file not mounted in this environment")
    strip = lambda l: l.strip().lstrip("- ").strip()
    env = {l.split("=", 1)[0]: l.split("=", 1)[1]
           for l in (strip(x) for x in open(compose, encoding="utf-8"))
           if "=" in l and strip(l).startswith("CAMERA_PAUSE_")}
    assert env.get("CAMERA_PAUSE_MIN_SPEECH_FRAMES_LIVINGROOM") == "3", env
    assert env.get("CAMERA_PAUSE_RUN_FRAMES_LIVINGROOM") == "10", env


def test_compose_has_no_alternate_wake_word_after_the_ab():
    """26.11.2026 final state: the temporary swap proved the Dobbi PATH
    (pip=0, ack=1) was the problem, not the word. Alternate word removed in
    every room; pause endpoint stays relaxed in the living room."""
    import os
    compose = "/ai-prod/docker-compose.yml"
    if not os.path.exists(compose):
        pytest.skip("compose file not mounted in this environment")
    lines = [l.strip().lstrip("- ").strip()
             for l in open(compose, encoding="utf-8")]
    alt = [l for l in lines
           if l.startswith(("WAKE_ALT_WORD_", "WAKE_ALT_TOKENS_"))]
    assert alt == [], alt
    assert not [l for l in lines if l.startswith("WAKE_WORD_LIVINGROOM=")], lines
    assert "CAMERA_PAUSE_MIN_SPEECH_FRAMES_LIVINGROOM=3" in lines, lines
    assert "CAMERA_PAUSE_RUN_FRAMES_LIVINGROOM=10" in lines, lines


def test_domovoy_fires_in_the_production_feed_with_a_command():
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOMOVOY)
    for text in ("домовой", "домовой включи свет", "домовой выключи свет",
                 "домовой какая погода на улице"):
        assert vosk_wake.transcript_has_wake(text, tokens) is True, text
    m = _matcher()
    m._rec.partial = "домовой включи свет"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"


def test_dobbi_fires_with_the_canonical_spelling_only():
    """The exact «домовой» token fires; empty allowlist leaves the old path alone."""
    m = _matcher()
    m._rec.partial = "домовой"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"
    m2 = _matcher(alt=())
    m2._rec.partial = "домовой"
    assert m2.feed(CHUNK) is False


@pytest.mark.asyncio
async def test_dobbi_ack_phrases_are_in_character():
    """All spoken lines for the second address must sound like the домовёнок:
    loyal and helpful, with his own turns from «Гарри Поттера»."""
    from camera_client import DOBBI_ACK_PHRASES

    s = _wake_session()
    with (
        patch.object(s, "_play_attention", new=AsyncMock()),
        patch.object(s, "_arm_wake_ack"),
    ):
        await s._fire_wake("vosk", kind="dobbi")
    assert set(s._wake_ack_pool) == set(DOBBI_ACK_PHRASES)
    assert all("?" not in p or p.endswith("?") for p in DOBBI_ACK_PHRASES)
    assert any("Добби" in p for p in DOBBI_ACK_PHRASES)


def test_wake_token_strip_handles_both_words_and_keeps_ordinary_words():
    """«Добби Добби» is TWO addresses and no command. The old strip only knew
    the hard-coded «компьютер», so it dispatched the pair to L2 and the room
    answered a monologue (24.11.2026 12:33 MSK+3)."""
    s = CameraSession.__new__(CameraSession)
    s.wake_keyword = "компьютер"
    s._wake_tokens = frozenset({"компьютер", "комп", "домовой"})
    s._wake_kind = "dobbi"
    assert s._strip_wake_words("Домовой Домовой") == ""
    assert s._strip_wake_words("Домовой, включи свет") == "включи свет"
    assert s._strip_wake_words("компьютер выключи свет") == "выключи свет"
    assert s._strip_wake_words("компот") == "компот", "a short token is not a prefix"
    # Old Dobbi mangles are ordinary words now and must not be stripped.
    assert s._strip_wake_words("Тоби выключите свет") == "Тоби выключите свет"


@pytest.mark.asyncio
async def test_a_double_address_is_not_dispatched_as_a_command():
    import time
    s = CameraSession(CameraConfig(stream_name="livingroom"))
    s.wake_keyword = "компьютер"
    s._wake_tokens = frozenset({"компьютер", "комп", "домовой"})
    s._wake_detected = True
    s._wake_only_pending = False
    s._last_wake_fired_at = time.time() - 0.7
    dispatched = []

    async def _fake(*a, **kw):
        dispatched.append(a)

    s._call_nanobot = _fake
    await s._handle_wake_or_command("Домовой Домовой", "camera")
    assert dispatched == [], "the wake word pair reached the router"
    assert s._wake_only_pending is True, "the room must wait for a real command"


class _FinalRec:
    def __init__(self, text, final):
        self.text, self.final = text, final

    def AcceptWaveform(self, pcm):
        return self.final

    def Result(self):
        return json.dumps({"text": self.text})

    def PartialResult(self):
        return json.dumps({"partial": self.text})


def test_a_late_final_of_the_same_word_does_not_refire():
    """The >3 s re-arm used to zero `_wake_hits` even when the FINAL's own text
    still held the SAME word, so `hits(1) > _wake_hits(0)` fired a second time
    for one spoken word. `camera._fire_wake()` then wiped the command being
    collected and the room greeted instead of answering."""
    m = _matcher()
    m._rec = _FinalRec("домовой", False)
    assert m.feed(CHUNK) is True
    m._last_fire_wall -= 5.0          # well past the same-breath debounce
    m._rec = _FinalRec("домовой", True)  # the SAME word's tail, finalised late
    assert m.feed(CHUNK) is False
    assert m.triggers == 1


def test_a_new_word_in_a_final_segment_still_fires_after_the_debounce():
    """`blocked=30`: a genuinely REPEATED, separate utterance must still fire.
    Unlike the same word's late tail it carries fresh loud audio after the
    debounce — that is the evidence the re-arm is allowed to act on."""
    m = _matcher()
    m._rec = _FinalRec("домовой", False)
    assert m.feed(CHUNK) is True
    m._last_fire_wall -= 5.0
    m.feed(LOUD_CHUNK)                # a new utterance after the debounce
    m._rec = _FinalRec("домовой", True)  # Vosk closed that new segment
    assert m.feed(CHUNK) is True
    assert m.triggers == 2


def test_a_same_breath_final_is_still_refused():
    """No time has passed and no fresh audio arrived: the final is the tail of
    the word that just fired, never a second wake."""
    m = _matcher()
    m._rec = _FinalRec("домовой", False)
    assert m.feed(CHUNK) is True
    m._rec = _FinalRec("домовой", True)
    assert m.feed(CHUNK) is False
    assert m.triggers == 1


def test_a_suppressed_repeat_is_counted_not_silent():
    """`hits <= _wake_hits` used to return False with no trace at all — the
    most expensive silence in the wake path. The counter goes into `vosk diag`.
    """
    m = _matcher()
    m._rec.partial = "домовой"
    assert m.feed(CHUNK) is True
    assert m.feed(CHUNK) is False
    assert m.blocked_same == 1


def test_back_to_wake_rearms_the_decoder_after_a_turn():
    """A recogniser left holding the previous word can silently block the next
    one: `hits <= _wake_hits` returns False with NO log line at all. Reset it
    when the turn ends, long after the same-breath window has passed."""
    s = _wake_session()

    class _W:
        def __init__(self):
            self.resets = 0
            self.begins = 0

        def reset(self):
            self.resets += 1

        def begin(self):
            self.begins += 1

    s._vosk_wake = _W()
    s._back_to_wake()
    assert s._vosk_wake.resets == 1
    assert s._vosk_wake.begins == 1


def test_back_to_wake_forgets_the_address_word():
    """Otherwise a follow-up to a Добби turn would keep using the dev brain."""
    s = _wake_session()
    s._wake_kind = "dobbi"
    s._wake_detected = True
    s._followup_only = True
    s._back_to_wake()
    assert s._wake_kind == "computer"


@pytest.mark.asyncio
async def test_dev_backend_is_used_only_for_dobbi():
    """Phase C seam: the wake word selects the brain, not the room or the text."""
    class _Backend:
        def __init__(self, name):
            self.name = name
            self.calls = []

        async def generate_response(self, **kwargs):
            self.calls.append(kwargs["text"])

    s = CameraSession(CameraConfig(stream_name="livingroom"))
    main_b, dev_b = _Backend("main"), _Backend("dev")
    s.backend, s.dev_backend = main_b, dev_b

    async def _no_player(_q):
        return None

    s._nanobot_player_task = _no_player
    s._wake_cmd_sent = False
    s._ww_base_thresh = 0.3
    s._ww_thresh = 0.3
    import camera_client
    camera_client.GLOBAL_TTS_UNTIL = 0.0   # another test may have left a guard

    s._wake_kind = "computer"
    await s._call_backend("включи свет")
    assert main_b.calls == ["включи свет"] and dev_b.calls == []

    s._wake_kind = "dobbi"
    await s._call_backend("включи свет")
    assert dev_b.calls == ["включи свет"]
    assert len(main_b.calls) == 1, "the normal backend must not see a pilot turn"


@pytest.mark.asyncio
async def test_dev_backend_env_builds_a_second_brain(monkeypatch):
    """The env must be filled through start_camera_sessions(), not just declared."""
    import main as main_mod

    created = []
    built = []

    class _FakeSession:
        def __init__(self, config=None, backend=None, dev_backend=None):
            self.config = config
            self.backend = backend
            self.dev_backend = dev_backend
            created.append(self)

        async def start(self):
            return None

    class _StubBackend:
        pass

    class _FakeClientSession:
        def __init__(self, timeout=None):
            pass

    monkeypatch.delenv("DISABLE_CAMERAS", raising=False)
    monkeypatch.setenv("CAMERA_STREAMS", "livingroom")
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(main_mod, "LLM_BACKEND_DOBBI", "hermes")
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)

    def _factory(name):
        built.append(name)
        return _StubBackend()

    monkeypatch.setattr(main_mod, "_build_llm_backend", _factory)
    await main_mod.start_camera_sessions()
    assert built == ["hermes"], "the primary brain must be the shared global"
    assert created[0].backend is main_mod.llm_backend
    assert isinstance(created[0].dev_backend, _StubBackend)


@pytest.mark.asyncio
async def test_no_dev_backend_keeps_the_old_path(monkeypatch):
    """Empty LLM_BACKEND_DOBBI must change nothing for either word."""
    import main as main_mod

    created = []

    class _FakeSession:
        def __init__(self, config=None, backend=None, dev_backend=None):
            self.config = config
            self.backend = backend
            self.dev_backend = dev_backend
            created.append(self)

        async def start(self):
            return None

    class _StubBackend:
        pass

    class _FakeClientSession:
        def __init__(self, timeout=None):
            pass

    monkeypatch.delenv("DISABLE_CAMERAS", raising=False)
    monkeypatch.setenv("CAMERA_STREAMS", "livingroom")
    monkeypatch.delenv("LLM_BACKEND_DOBBI", raising=False)
    # The constant is read at import; the env alone does not reset it.
    monkeypatch.setattr(main_mod, "LLM_BACKEND_DOBBI", "")
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(main_mod.backends, "CascadeBackend",
                        lambda *a, **k: _StubBackend())
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)

    await main_mod.start_camera_sessions()
    assert created[0].dev_backend is None


@pytest.mark.asyncio
async def test_dobbi_turn_synthesizes_with_aidar_but_computer_stays_on_baya():
    """The second address gets its own TTS voice — but only while that turn is
    open, and only while Silero is the active backend."""
    import numpy as np
    import camera_client

    raw = (np.sin(np.linspace(0, 40, 4800)) * 17000).astype(np.int16).tobytes()

    class _Resp:
        status = 200

        async def read(self):
            return b"mp3"

    class _Post:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Http:
        def __init__(self):
            self.payloads = []

        def post(self, _url, json=None, **_kw):
            self.payloads.append(dict(json or {}))
            return _Post()

    class _Seg:
        raw_data = raw

        def set_frame_rate(self, _r):
            return self

        def set_channels(self, _c):
            return self

        def set_sample_width(self, _w):
            return self

    s = CameraSession.__new__(CameraSession)
    s.stream_name = "livingroom"
    s.tts_url = "http://192.168.22.102:5053/v1/audio/speech"
    s.tts_voice = "baya"
    s.tts_api_key = ""
    s.tts_backend = "silero"
    s._tts_voice_dobbi = "aidar"
    s._tts_target_peak = 20000
    s._wake_kind = "computer"
    s.http_session = _Http()

    with patch.object(camera_client.AudioSegment, "from_file", return_value=_Seg()):
        await s._tts_fetch("Включаю.")
        s._wake_kind = "dobbi"
        await s._tts_fetch("Включаю.")
    assert [p["voice"] for p in s.http_session.payloads] == ["baya", "aidar"]


def test_dobbi_voice_is_ignored_when_the_backend_is_not_silero():
    s = CameraSession.__new__(CameraSession)
    s.tts_url = "http://192.168.22.102:5052/v1/audio/speech"
    s.tts_voice = "Zofija Kendrick"
    s.tts_backend = "coqui"
    s._tts_voice_dobbi = "aidar"
    s._wake_kind = "dobbi"
    assert s._tts_voice_for_turn() == "Zofija Kendrick"


@pytest.mark.asyncio
async def test_swapped_wake_words_are_wired_per_room(monkeypatch):
    """Temporary A/B 26.11: «домовой» becomes the PRIMARY word in the living
    room and «компьютер» the alternate. Needs a per-room WAKE_WORD override and
    acceptance of `computer` as an alt-word key — a config-only test cannot see
    a missing fill in start_camera_sessions()."""
    import main as main_mod

    created = []

    class _FakeSession:
        def __init__(self, config=None, backend=None, dev_backend=None):
            self.config = config
            created.append(self)

        async def start(self):
            return None

    class _StubBackend:
        pass

    class _FakeClientSession:
        def __init__(self, timeout=None):
            pass

    monkeypatch.delenv("DISABLE_CAMERAS", raising=False)
    monkeypatch.setenv("CAMERA_STREAMS", "livingroom")
    monkeypatch.setenv("WAKE_WORD_LIVINGROOM", "домовой")
    monkeypatch.setenv("WAKE_ALT_WORD_LIVINGROOM", "computer")
    monkeypatch.setenv("WAKE_ALT_TOKENS_LIVINGROOM", "компьютер|комп|домовой")
    monkeypatch.delenv("CAMERA_WAKE_LOUD_BAR_LIVINGROOM", raising=False)
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(main_mod.backends, "CascadeBackend",
                        lambda *a, **k: _StubBackend())
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)

    await main_mod.start_camera_sessions()
    cfg = created[0].config
    assert cfg.wake_keyword == "домовой"
    assert cfg.alt_wake_word == "computer"
    assert cfg.alt_wake_tokens == "компьютер|комп|домовой"


@pytest.mark.asyncio
async def test_camera_config_carries_the_alt_wake_env(monkeypatch):
    """The same wiring trap as TTS_VOICE: a new field must be FILLED, not just
    declared on CameraConfig — a config-only test cannot see a missing fill."""
    import main as main_mod

    created = []

    class _FakeSession:
        def __init__(self, config=None, backend=None, dev_backend=None):
            self.config = config
            created.append(self)

        async def start(self):
            return None

    class _StubBackend:
        pass

    class _FakeClientSession:
        def __init__(self, timeout=None):
            pass

    monkeypatch.delenv("DISABLE_CAMERAS", raising=False)
    monkeypatch.setenv("CAMERA_STREAMS", "livingroom")
    monkeypatch.setenv("WAKE_ALT_WORD_LIVINGROOM", "dobbi")
    monkeypatch.setenv("WAKE_ALT_TOKENS_LIVINGROOM", "домовой")
    monkeypatch.delenv("CAMERA_WAKE_LOUD_BAR_LIVINGROOM", raising=False)
    monkeypatch.setenv("TTS_VOICE_DOBBI", "aidar")
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(main_mod.backends, "CascadeBackend",
                        lambda *a, **k: _StubBackend())
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)

    await main_mod.start_camera_sessions()
    cfg = created[0].config
    assert cfg.alt_wake_word == "dobbi"
    assert cfg.alt_wake_tokens == "домовой"
    assert cfg.tts_voice_dobbi == "aidar"
    assert cfg.wake_loud_bar == 0, "the small model keeps its per-word bars"
    assert CameraConfig(stream_name="x").wake_loud_bar == 0
    assert CameraConfig(stream_name="x").alt_wake_word == ""


# --- Домовой (26.11.2026): exact token, old Dobbi mis-hearings are gone -------

def test_domovoy_is_an_exact_token_and_old_dobbi_forms_are_gone():
    alt = frozenset(vosk_wake.DOMOVOY_TOKENS)
    assert vosk_wake.transcript_has_wake(
        "домовой включи свет", vosk_wake.WAKE_TOKENS | alt) is True
    for old in ("добрый включи свет", "дарби включи свет", "да включи свет",
                "доме какая погода", "давай без включи свет"):
        assert vosk_wake.transcript_has_wake(
            old, vosk_wake.WAKE_TOKENS | alt) is False, old


def test_domovoy_wake_kind_stays_dobbi():
    alt = frozenset(vosk_wake.DOMOVOY_TOKENS)
    assert vosk_wake.wake_kind_in("домовой включи свет", alt, "dobbi") == "dobbi"
    assert vosk_wake.wake_kind_in(
        "домовой компьютер включи свет", alt, "dobbi") == "computer"


