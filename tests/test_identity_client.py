"""Gateway identity client: 30 s cache, one Bearer header, loud failures."""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identity_client  # noqa: E402


class Resp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class Stub:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []
        self.now = [0.0]

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw.get("headers")))
        return Resp(*self.payloads.get(url, (200, {})))

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw.get("headers")))
        return Resp(*self.payloads.get(url, (200, {})))


PROFILES = {"profiles": [
    {"key": "alexander", "display_name": "Александр", "honorific": "сэр",
     "telegram_chat_ids": ["543867556"], "voice_enrolled": True},
    {"key": "marina", "display_name": "Марина", "honorific": "мэм",
     "telegram_chat_ids": ["1620243752"], "voice_enrolled": False},
]}


def make(payloads=None, ttl=30.0):
    stub = Stub(payloads or {"http://id/api/profiles": (200, PROFILES)})
    client = identity_client.IdentityClient(
        "http://id", "T", ttl=ttl, session=stub, clock=lambda: stub.now[0])
    return client, stub


def test_empty_url_keeps_identity_none(monkeypatch):
    monkeypatch.setenv("IDENTITY_URL", "")
    identity_client.init_identity()
    assert identity_client.IDENTITY is None
    monkeypatch.setenv("IDENTITY_URL", "http://id")
    monkeypatch.setenv("IDENTITY_TOKEN", "T")
    identity_client.init_identity()
    assert isinstance(identity_client.IDENTITY, identity_client.IdentityClient)
    identity_client.IDENTITY = None


@pytest.mark.asyncio
async def test_cache_refreshes_after_ttl():
    client, stub = make(ttl=30.0)
    first = await client.profiles()
    await client.profiles()
    assert len(stub.calls) == 1
    stub.now[0] = 31.0
    await client.profiles()
    assert len(stub.calls) == 2
    assert first["alexander"]["honorific"] == "сэр"
    assert stub.calls[0][2]["Authorization"] == "Bearer T"


@pytest.mark.asyncio
async def test_identity_failure_raises_and_callers_fall_back():
    client, _ = make({"http://id/api/identify": (503, {"detail": "down"})})
    with pytest.raises(identity_client.IdentityError) as exc:
        await client.identify(b"ogg")
    assert exc.value.status == 503


@pytest.mark.asyncio
async def test_unknown_person_has_no_honorific():
    client, _ = make({"http://id/api/identify": (
        200, {"person": "testuser", "honorific": "", "display_name": "testuser",
              "confidence": 0.42})})
    person, honorific, conf = await client.identify(b"ogg")
    assert (person, honorific) == ("testuser", "")
    assert conf == 0.42
    assert client.honorific_for(person) == ""


@pytest.mark.asyncio
async def test_identify_surfaces_confidence_for_callers_to_floor():
    """The identity service returns confidence; callers must be able to apply
    the same > 0.1 floor the direct SpeakerID path already uses."""
    client, _ = make({"http://id/api/identify": (
        200, {"person": "alexander", "honorific": "сэр", "confidence": 0.87})})
    person, honorific, conf = await client.identify(b"ogg")
    assert (person, honorific, conf) == ("alexander", "сэр", 0.87)


@pytest.mark.asyncio
async def test_chat_map_from_cached_profiles():
    client, _ = make()
    assert client.chat_map() == {}
    await client.profiles()
    assert client.chat_map() == {
        "543867556": "alexander", "1620243752": "marina"}
