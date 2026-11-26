"""UI profile edits reach Telegram without recreating the gateway."""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identity_client  # noqa: E402
import main  # noqa: E402


class Resp:
    def __init__(self, payload):
        self.status = 200
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self.payload


class Stub:
    def __init__(self, profiles):
        self.profiles = profiles
        self.gets = 0

    def get(self, url, **kw):
        self.gets += 1
        return Resp({"profiles": self.profiles})


def profiles(honorific):
    return [{"key": "alexander", "honorific": honorific,
             "telegram_chat_ids": ["543867556"]}]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(identity_client, "IDENTITY", None)
    yield
    identity_client.IDENTITY = None


@pytest.mark.asyncio
async def test_honorific_comes_from_the_identity_profile_not_the_import_map(monkeypatch):
    stub = Stub(profiles("сэр"))
    client = identity_client.IdentityClient("http://id", "T", ttl=0.01, session=stub)
    monkeypatch.setattr(identity_client, "IDENTITY", client)
    await client.profiles()
    assert main.honorific_for_chat("543867556") == "сэр"
    stub.profiles = profiles("мэм")
    await asyncio.sleep(0.02)
    await client.profiles()
    assert stub.gets == 2
    assert main.honorific_for_chat("543867556") == "мэм"


@pytest.mark.asyncio
async def test_refresh_loop_keeps_the_cache_current(monkeypatch):
    stub = Stub(profiles("сэр"))
    client = identity_client.IdentityClient("http://id", "T", ttl=0.0, session=stub)
    monkeypatch.setattr(identity_client, "IDENTITY", client)
    task = asyncio.create_task(main.identity_refresh_loop(interval=0.01))
    for _ in range(30):
        await asyncio.sleep(0.01)
        if stub.gets >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stub.gets >= 3
