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
DOBBI = ("добби", "доби")


class _ScriptedRec:
    def __init__(self, partial=""):
        self.partial = partial

    def AcceptWaveform(self, pcm):
        return False

    def Result(self):
        return json.dumps({"text": self.partial})

    def PartialResult(self):
        return json.dumps({"partial": self.partial})


def _matcher(alt=DOBBI):
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
    m._rec.partial = "добби"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"

    m.reset()
    m._rec = _ScriptedRec("компьютер")
    assert m.feed(CHUNK) is True
    assert m.last_kind == "computer"


def test_one_breath_with_both_words_fires_once_and_kind_is_the_last_word():
    m = _matcher()
    m._rec.partial = "компьютер добби"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi", "the word closest to the command decides"
    assert m.feed(CHUNK) is False, "the tail of one breath must not fire again"
    assert m.triggers == 1


def test_a_second_word_appended_after_a_fire_is_not_a_new_wake():
    """The hypothesis GROWS. «компьютер» then «компьютер добби» must not fire
    twice — the second fire would cancel the command being collected."""
    m = _matcher()
    m._rec.partial = "компьютер"
    assert m.feed(CHUNK) is True
    m._rec.partial = "компьютер добби"
    assert m.feed(CHUNK) is False
    # The words leave the hypothesis: the next address is a new one.
    m._rec.partial = ""
    assert m.feed(CHUNK) is False
    m._rec.partial = "добби"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "dobbi"


def test_dobbi_does_not_match_when_its_allowlist_is_empty():
    m = _matcher(alt=())
    m._rec.partial = "добби"
    assert m.feed(CHUNK) is False, "empty alt allowlist must leave the old path alone"
    m._rec.partial = "компьютер"
    assert m.feed(CHUNK) is True
    assert m.last_kind == "computer"


def test_a_misspelling_of_the_second_word_is_a_diagnostic_not_a_match():
    tokens = vosk_wake.WAKE_TOKENS | frozenset(DOBBI)
    assert vosk_wake.transcript_has_wake("доббе", tokens) is False
    assert vosk_wake.wake_near_miss("доббе", tokens) == "доббе"


def test_wake_kind_in_uses_the_last_word_in_the_hypothesis():
    fn = vosk_wake.wake_kind_in
    assert fn("компьютер добби", frozenset(DOBBI)) == "dobbi"
    assert fn("добби компьютер", frozenset(DOBBI)) == "computer"
    assert fn("включи свет", frozenset(DOBBI)) == ""


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

def test_back_to_wake_forgets_the_address_word():
    """Otherwise a follow-up to a Dobbi turn would keep using the dev brain."""
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
    monkeypatch.setenv("WAKE_ALT_TOKENS_LIVINGROOM", "добби|доби")
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
    assert cfg.alt_wake_tokens == "добби|доби"
    assert CameraConfig(stream_name="x").alt_wake_word == ""
