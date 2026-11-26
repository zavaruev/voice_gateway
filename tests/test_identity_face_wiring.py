"""Face + Telegram read identity too; Frigate detection stays on the face API."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import face_events  # noqa: E402
import frigate_events  # noqa: E402
import identity_client  # noqa: E402
import main  # noqa: E402


class FakeIdentity:
    def __init__(self):
        self.chats = {"543867556": "alexander"}
        self.faces = [{"name": "alexander", "confidence": 0.8}]
        self.profiles_calls = 0

    async def profiles(self):
        self.profiles_calls += 1
        return {
            "alexander": {"key": "alexander", "honorific": "сэр"},
            "marina": {"key": "marina", "honorific": "мэм"},
        }

    async def recognize(self, image, threshold):
        return self.faces

    def chat_map(self):
        return dict(self.chats)

    def honorific_for(self, person):
        return {"alexander": "сэр", "marina": "мэм"}.get(str(person).lower(), "")


class Resp:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return {"faces": [{"name": "alexander", "confidence": 0.9}]}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(url)
        return Resp()


@pytest.fixture(autouse=True)
def clean_identity(monkeypatch):
    monkeypatch.setattr(identity_client, "IDENTITY", None)
    monkeypatch.setattr(main, "FACE_GREETING_TOKEN", "x")
    monkeypatch.setattr(main.face_greeting, "warm", lambda names: _record(names))
    yield


RECORDED = []


def _record(names):
    RECORDED.append(list(names))


@pytest.mark.asyncio
async def test_greeting_roster_comes_from_identity(monkeypatch):
    fake = FakeIdentity()
    monkeypatch.setattr(identity_client, "IDENTITY", fake)
    RECORDED.clear()
    await main._warm_face_greetings()
    assert RECORDED == [["alexander", "marina"]]


@pytest.mark.asyncio
async def test_recognize_uses_identity_when_set(tmp_path, monkeypatch):
    fake = FakeIdentity()
    monkeypatch.setattr(identity_client, "IDENTITY", fake)
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"jpg")
    sess = FakeSession()
    watcher = face_events.FaceGreetingWatcher.__new__(face_events.FaceGreetingWatcher)
    got = await watcher._recognize_frame(sess, str(frame), 0.35)
    assert got == fake.faces
    assert sess.calls == []


@pytest.mark.asyncio
async def test_frigate_detect_stays_direct():
    sess = FakeSession()
    watcher = frigate_events.FrigateEventInjector.__new__(frigate_events.FrigateEventInjector)
    await watcher._detect(sess, b"jpg")
    assert sess.calls == [f"{frigate_events.FACE_API}/detect"]


def test_telegram_honorific_reflects_ui_edit(monkeypatch):
    fake = FakeIdentity()
    monkeypatch.setattr(identity_client, "IDENTITY", fake)
    assert main.honorific_for_chat("543867556") == "сэр"
    fake.chats["543867556"] = "marina"
    assert main.honorific_for_chat("543867556") == "мэм"
