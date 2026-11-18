"""Tests for Frigate person-snapshot injection into an active face event."""

import json
import time

import pytest

import frigate_events as fe


def test_active_event_ignores_stale_days_and_frigate_frames(tmp_path):
    day = tmp_path / "2026-11-18"
    day.mkdir()
    frame = day / "12-00-00_camera_entrance1_face01.jpg"
    frame.write_bytes(b"x")
    injected = day / "12-00-00_camera_entrance1_face90_frigate.jpg"
    injected.write_bytes(b"y")
    now = time.time()
    # Make the injected frame newer — if it counted, the event would look
    # freshly active even though no real burst is running.
    import os
    os.utime(injected, (now, now))
    os.utime(frame, (now - 10, now - 10))
    assert fe.active_event(str(tmp_path), now) == ("2026-11-18", "12-00-00")
    os.utime(frame, (now - 200, now - 200))
    assert fe.active_event(str(tmp_path), now) is None


@pytest.mark.asyncio
async def test_poll_once_injects_a_detected_face_into_the_active_event(
        tmp_path, monkeypatch):
    day = tmp_path / "2026-11-18"
    day.mkdir()
    frame = day / "12-00-00_camera_entrance1_face01.jpg"
    frame.write_bytes(b"x")
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "FACE_API", "http://face.test")
    monkeypatch.setattr(fe, "FRIGATE_URL", "http://frigate.test")

    ev = {"id": "evt1", "camera": "entrance1", "start_time": time.time(),
          "has_snapshot": True}

    class Resp:
        def __init__(self, body=None, raw=b""):
            self.status = 200
            self._body = body
            self._raw = raw

        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self): return self._body
        async def read(self): return self._raw

    class Sess:
        def get(self, url, **kw):
            if url.endswith("/api/events"):
                return Resp([ev])
            return Resp(raw=b"JPEG")

        def post(self, url, **kw):
            return Resp({"faces": [{"face_size": 220, "det_score": 0.74}]})

    inj = fe.FrigateEventInjector()
    assert await inj.poll_once(Sess()) is True
    injected = day / "12-00-00_camera_entrance1_face90_frigate.jpg"
    assert injected.read_bytes() == b"JPEG"
    verdict = json.loads((day / "12-00-00_frames.jsonl").read_text().strip())
    assert verdict["quality_ok"] and verdict["face_size"] == 220
    meta = json.loads((day / "12-00-00_meta.json").read_text())
    assert meta["frame_details"][0]["path"] == str(injected)
    # The same event is not injected twice on the next poll.
    assert await inj.poll_once(Sess()) is False


@pytest.mark.asyncio
async def test_no_active_event_means_no_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    inj = fe.FrigateEventInjector()

    class Sess:
        def get(self, url, **kw):
            class R:
                status = 200
                async def __aenter__(self): return self
                async def __aexit__(self, *a): return False
                async def json(self): return []
            return R()

    assert await inj.poll_once(Sess()) is False
