"""Both speaker-ID call sites use identity, with a loud direct fallback."""
import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_client  # noqa: E402
import identity_client  # noqa: E402
import main  # noqa: E402


class FakeIdentity:
    def __init__(self, result=("alexander", "сэр", 0.9), error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def identify(self, audio):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeProc:
    def __await__(self):
        async def done():
            return self
        return done().__await__()

    async def communicate(self, data=None):
        return b"ogg", b""


class FakeResp:
    status = 200

    def __init__(self, uid="alexander"):
        self.uid = uid

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return {"user_id": self.uid, "confidence": 0.9}


class FakeSession:
    def __init__(self, uid="alexander"):
        self.calls = []
        self.uid = uid

    def post(self, url, **kw):
        self.calls.append(url)
        return FakeResp(self.uid)


def cam(**attrs):
    s = camera_client.CameraSession.__new__(camera_client.CameraSession)
    s.stream_name = "livingroom"
    s._speaker_id_url = "http://spk/identify"
    s.http_session = FakeSession()
    s._last_uid = "unknown"
    s._honorific_map = {"alexander": "сэр"}
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


@pytest.fixture(autouse=True)
def clean_identity(monkeypatch):
    monkeypatch.setattr(identity_client, "IDENTITY", None)
    monkeypatch.setattr(main, "_last_honorific", "")
    yield


@pytest.mark.asyncio
async def test_camera_identity_honorific_wins(monkeypatch):
    fake = FakeIdentity()
    monkeypatch.setattr(identity_client, "IDENTITY", fake)
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    s = cam()
    assert await s._fetch_speaker_id(b"wav") == "alexander"
    assert fake.calls == 1
    assert s._last_honorific == "сэр"
    assert s._current_honorific() == "сэр"
    assert s.http_session.calls == []


@pytest.mark.asyncio
async def test_esp32_identity_honorific_wins(monkeypatch):
    fake = FakeIdentity()
    monkeypatch.setattr(identity_client, "IDENTITY", fake)
    sess = FakeSession()
    assert await main.fetch_speaker_id(b"ogg", sess) == "alexander"
    assert fake.calls == 1
    assert main._last_honorific == "сэр"
    assert sess.calls == []


@pytest.mark.asyncio
async def test_identity_failure_falls_back_to_direct_engine(monkeypatch):
    monkeypatch.setattr(identity_client, "IDENTITY",
                        FakeIdentity(error=identity_client.IdentityError(503, "down")))
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    s = cam()
    uid = await s._fetch_speaker_id(b"wav")
    assert uid == "alexander"
    assert s.http_session.calls == ["http://spk/identify"]
    s._last_uid = uid   # the caller does this on utterance end
    assert s._current_honorific() == "сэр"   # local map fallback


@pytest.mark.asyncio
async def test_ambient_utterance_does_not_call_speaker_id(monkeypatch):
    """Before Phase B ambient speaker-ID ran ~50x/min, and identity + direct
    fallback doubled it past the engine's 10/60s limit — a permanent 429."""
    from unittest.mock import AsyncMock
    s = cam()
    s._wake_detected = False
    s._wake_expires = 0.0
    s._fetch_speaker_id = AsyncMock(return_value="unknown")
    await s._log_speaker_id_full_utterance(b"x" * 4000)
    s._fetch_speaker_id.assert_not_awaited()
    s._wake_detected = True
    await s._log_speaker_id_full_utterance(b"x" * 4000)
    s._fetch_speaker_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_identity_429_does_not_double_the_engine_load(monkeypatch):
    monkeypatch.setattr(identity_client, "IDENTITY", FakeIdentity(
        error=identity_client.IdentityError(429, "Too Many Requests")))
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    s = cam()
    assert await s._fetch_speaker_id(b"wav") == "unknown"
    assert s.http_session.calls == []


def test_hand_built_sessions_carry_honorific_fields():
    s = camera_client.CameraSession.__new__(camera_client.CameraSession)
    s._last_uid = "alexander"
    s._honorific_map = {"alexander": "сэр"}
    assert s._current_honorific() == "сэр"  # class default, no AttributeError


@pytest.mark.asyncio
async def test_stale_identity_honorific_does_not_survive_an_outage(monkeypatch):
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    s = cam()
    monkeypatch.setattr(identity_client, "IDENTITY", FakeIdentity())
    assert await s._fetch_speaker_id(b"wav") == "alexander"
    assert s._current_honorific() == "сэр"
    # Identity dies; the direct engine now identifies a DIFFERENT person.
    monkeypatch.setattr(identity_client, "IDENTITY",
                        FakeIdentity(error=identity_client.IdentityError(503, "down")))
    s.http_session = FakeSession(uid="marina")
    s._honorific_map["marina"] = "мэм"
    assert await s._fetch_speaker_id(b"wav") == "marina"
    s._last_uid = "marina"
    assert s._current_honorific() == "мэм"


@pytest.mark.asyncio
async def test_empty_honorific_from_identity_is_authoritative(monkeypatch):
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    monkeypatch.setattr(identity_client, "IDENTITY",
                        FakeIdentity(result=("alexander", "", 0.9)))
    s = cam(_last_uid="alexander")
    await s._fetch_speaker_id(b"wav")
    assert s._current_honorific() == ""


# --- Fix 1: ESP32 dispatch must carry persona/honorific like camera/Telegram ---
@pytest.mark.asyncio
async def test_esp32_dispatch_passes_honorific_and_empty_persona(monkeypatch):
    """The camera passes persona/honorific to generate_response; the ESP32
    dispatch did neither, so the Dobbi styling never reached a satellite.

    ESP32 has no per-turn wake kind (it always uses the single primary
    llm_backend chosen by LLM_BACKEND), so persona is "" here — mirroring a
    camera turn whose _wake_kind != "dobbi".
    """
    seen = {}

    class StubBackend:
        async def generate_response(self, **kw):
            seen.update(kw)
            kw["response_queue"].put_nowait(None)

    async def noop_player(q, ws, state):
        return None

    async def noop_stop(ws, state):
        return None

    async def noop_finalize(ws, state, text):
        return False

    class StubWS:
        async def send_json(self, msg):
            return None

    monkeypatch.setattr(main, "llm_backend", StubBackend())
    monkeypatch.setattr(main, "_hermes_player_task", noop_player)
    monkeypatch.setattr(main, "send_tts_stop", noop_stop)
    monkeypatch.setattr(main, "_finalize_turn_followup", noop_finalize)
    monkeypatch.setattr(main, "make_chat_id", lambda mac: "chat-" + str(mac))
    monkeypatch.setattr(main, "_last_honorific", "сэр", raising=False)
    monkeypatch.setattr(main, "_honorific_from_identity", True, raising=False)
    state = {"mac": "AA:BB", "sid": "s1", "watchdog": None}
    await main._dispatch_hermes("включи свет", "alexander", state, StubWS(),
                                time.time(), time.time())
    assert seen.get("persona") == ""
    assert seen.get("honorific") == "сэр"


@pytest.mark.asyncio
async def test_esp32_honorific_flag_tracks_identity_success_and_failure(monkeypatch):
    """Mirror of camera_client: identity is authoritative only when it actually
    answered; after ANY failure the local map decides for the NEW engine uid."""
    monkeypatch.setattr(main, "_last_honorific", "", raising=False)
    monkeypatch.setattr(main, "_honorific_from_identity", False, raising=False)
    monkeypatch.setattr(main, "HONORIFIC_MAP", {"alexander": "сэр", "marina": "мэм"})
    monkeypatch.setattr(identity_client, "IDENTITY",
                        FakeIdentity(result=("alexander", "сэр", 0.9)))
    sess = FakeSession()
    assert await main.fetch_speaker_id(b"ogg", sess) == "alexander"
    assert main._honorific_from_identity is True
    assert main._current_honorific("alexander") == "сэр"

    # Identity dies (503). The direct engine now identifies a DIFFERENT person.
    monkeypatch.setattr(identity_client, "IDENTITY",
                        FakeIdentity(error=identity_client.IdentityError(503, "down")))
    sess2 = FakeSession(uid="marina")
    assert await main.fetch_speaker_id(b"ogg", sess2) == "marina"
    assert main._honorific_from_identity is False
    assert main._current_honorific("marina") == "мэм"


# --- Fix 2: the identity confidence must use the same > 0.1 floor as the
# direct SpeakerID path, at BOTH call sites ---
@pytest.mark.asyncio
async def test_camera_rejects_low_confidence_identity_match(monkeypatch):
    """Identity may name somebody on 0.05 confidence — that is a guess, not a
    person. The camera must refuse it, clear any stale address, and NOT call
    the direct engine (identity already consumed the same audio)."""
    monkeypatch.setattr(camera_client.asyncio, "create_subprocess_exec",
                        lambda *a, **kw: FakeProc())
    monkeypatch.setattr(identity_client, "IDENTITY", FakeIdentity(
        result=("alexander", "сэр", 0.05)))
    s = cam()
    s._last_honorific = "сэр"
    s._honorific_from_identity = True
    assert await s._fetch_speaker_id(b"wav") == "unknown"
    assert s._last_honorific == ""
    assert s._honorific_from_identity is False
    assert s.http_session.calls == []


@pytest.mark.asyncio
async def test_esp32_identity_429_does_not_double_the_engine_load(monkeypatch):
    """The camera refuses a direct retry on 429; the ESP32 path must too.
    The engine rate-limits 10/60s, so a second call only deepens the outage."""
    monkeypatch.setattr(main, "_last_honorific", "сэр", raising=False)
    monkeypatch.setattr(main, "_honorific_from_identity", True, raising=False)
    monkeypatch.setattr(identity_client, "IDENTITY", FakeIdentity(
        error=identity_client.IdentityError(429, "Too Many Requests")))
    sess = FakeSession()
    assert await main.fetch_speaker_id(b"ogg", sess) == "unknown"
    assert sess.calls == []
    assert main._last_honorific == ""
    assert main._honorific_from_identity is False


@pytest.mark.asyncio
async def test_esp32_rejects_low_confidence_identity_match(monkeypatch):
    monkeypatch.setattr(main, "_last_honorific", "сэр", raising=False)
    monkeypatch.setattr(main, "_honorific_from_identity", True, raising=False)
    monkeypatch.setattr(identity_client, "IDENTITY", FakeIdentity(
        result=("alexander", "сэр", 0.05)))
    sess = FakeSession()
    assert await main.fetch_speaker_id(b"ogg", sess) == "unknown"
    assert main._last_honorific == ""
    assert main._honorific_from_identity is False
    assert sess.calls == []
