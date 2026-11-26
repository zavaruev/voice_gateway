"""Camera-audio capture for enrollment: WAV from the live session, no second consumer."""
import asyncio
import io
import os
import sys
import time
import wave

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_client  # noqa: E402
import main  # noqa: E402


class FakeSession:
    def __init__(self, name):
        self.stream_name = name
        self.seconds = []

    async def record_sample(self, seconds):
        self.seconds.append(seconds)
        return b"WAV" + str(seconds).encode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("IDENTITY_CAPTURE_TOKEN", "CAP")
    room = FakeSession("livingroom")
    monkeypatch.setattr(main, "_camera_sessions", [room])
    return TestClient(main.app), room


def auth():
    return {"Authorization": "Bearer CAP"}


def test_capture_requires_token(client):
    api, _ = client
    assert api.post("/api/identity/capture", json={"room": "livingroom"}).status_code == 401
    assert api.post("/api/identity/capture", json={"room": "livingroom"},
                    headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_capture_finds_session_by_room(client):
    api, room = client
    r = api.post("/api/identity/capture",
                 json={"room": "livingroom", "seconds": 6}, headers=auth())
    assert r.status_code == 200
    assert r.content == b"WAV6.0"
    assert room.seconds == [6.0]
    assert api.post("/api/identity/capture",
                    json={"room": "kitchen"}, headers=auth()).status_code == 404


def test_seconds_are_clamped_to_2_30(client):
    api, room = client
    api.post("/api/identity/capture", json={"room": "livingroom", "seconds": 0},
             headers=auth())
    api.post("/api/identity/capture", json={"room": "livingroom", "seconds": 100},
             headers=auth())
    assert room.seconds == [2.0, 30.0]


@pytest.mark.asyncio
async def test_a_second_capture_is_refused_not_overwritten():
    s = camera_client.CameraSession.__new__(camera_client.CameraSession)
    s.stream_name = "livingroom"
    task = asyncio.create_task(s.record_sample(2.0))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already active"):
        await s.record_sample(2.0)
    s._capture_deadline = time.monotonic() - 1
    await task


def test_busy_session_returns_409(monkeypatch):
    class Busy(FakeSession):
        async def record_sample(self, seconds):
            raise RuntimeError("capture already active on this room")
    monkeypatch.setenv("IDENTITY_CAPTURE_TOKEN", "CAP")
    monkeypatch.setattr(main, "_camera_sessions", [Busy("livingroom")])
    r = TestClient(main.app).post("/api/identity/capture",
                                  json={"room": "livingroom"}, headers=auth())
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_record_sample_writes_a_valid_wav():
    s = camera_client.CameraSession.__new__(camera_client.CameraSession)
    s.stream_name = "livingroom"
    task = asyncio.create_task(s.record_sample(2.0))
    await asyncio.sleep(0)
    assert s._capture_buf is not None
    for _ in range(4):
        s._tap_capture(b"\x00" * 2560)
    s._capture_deadline = time.monotonic() - 1
    wav = await task
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 4 * 1280
