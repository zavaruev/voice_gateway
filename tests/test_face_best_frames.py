import pytest

import face_events

MATCHES = [
    (0.72, "/frames/a.jpg", "camera_entrance1"),
    (0.91, "/frames/b.jpg", "camera_entrance1"),
    (0.68, "/frames/c.jpg", "camera_entrance2"),
]


class Poster:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def __call__(self, path, name):
        self.calls.append((path, name))
        if self.fail:
            raise RuntimeError("identity refused")
        return True


@pytest.mark.asyncio
async def test_flag_off_never_posts(monkeypatch):
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", False)
    face_events._LAST_CURATED.clear()
    poster = Poster()
    assert await face_events.maybe_curate_best("vladislava", MATCHES, poster=poster, now=1000) is False
    assert poster.calls == []


@pytest.mark.asyncio
async def test_best_matching_frame_is_posted_once_per_event(monkeypatch):
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    monkeypatch.setattr(face_events, "FACE_BEST_FRAME_MIN_GAP_S", 3600)
    face_events._LAST_CURATED.clear()
    poster = Poster()
    assert await face_events.maybe_curate_best("vladislava", MATCHES, poster=poster, now=1000) is True
    assert poster.calls == [("/frames/b.jpg", "vladislava")]


@pytest.mark.asyncio
async def test_hourly_gap_blocks_a_second_event(monkeypatch):
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    monkeypatch.setattr(face_events, "FACE_BEST_FRAME_MIN_GAP_S", 3600)
    face_events._LAST_CURATED.clear()
    poster = Poster()
    await face_events.maybe_curate_best("vladislava", MATCHES, poster=poster, now=1000)
    assert await face_events.maybe_curate_best("vladislava", MATCHES, poster=poster, now=2000) is False
    assert len(poster.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_post_does_not_start_the_gap(monkeypatch):
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    monkeypatch.setattr(face_events, "FACE_BEST_FRAME_MIN_GAP_S", 3600)
    face_events._LAST_CURATED.clear()
    poster = Poster(fail=True)
    assert await face_events.maybe_curate_best("vladislava", MATCHES, poster=poster, now=1000) is False
    assert "vladislava" not in face_events._LAST_CURATED


@pytest.mark.asyncio
async def test_no_matching_frames_is_a_noop(monkeypatch):
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    face_events._LAST_CURATED.clear()
    poster = Poster()
    assert await face_events.maybe_curate_best("vladislava", [], poster=poster, now=1000) is False
    assert poster.calls == []


class CuratorResponse:
    """Stand-in for the aiohttp response context manager used by post_best_frame."""

    def __init__(self, status=200, body=None):
        self.status = status
        self._body = {"status": "unchanged"} if body is None else body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._body


class RecordingSession:
    """Replace aiohttp.ClientSession: record the POST and return a fixed reply."""

    posts = []
    response = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        RecordingSession.posts.append(url)
        return RecordingSession.response


@pytest.mark.asyncio
async def test_unchanged_curator_is_a_handled_noop(monkeypatch, tmp_path):
    """A full gallery answers 200/unchanged: record the gap, do not re-POST."""
    frame = tmp_path / "face.jpg"
    frame.write_bytes(b"\xff\xd8fake")
    monkeypatch.setenv("IDENTITY_URL", "http://identity.test:18794")
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    monkeypatch.setattr(face_events, "FACE_BEST_FRAME_MIN_GAP_S", 3600)
    face_events._LAST_CURATED.clear()
    RecordingSession.posts = []
    RecordingSession.response = CuratorResponse()
    monkeypatch.setattr(face_events.aiohttp, "ClientSession", RecordingSession)
    matches = [(0.91, str(frame), "camera_entrance1")]

    assert await face_events.maybe_curate_best("alexander", matches, now=1000) is True
    assert RecordingSession.posts == [
        "http://identity.test:18794/api/profiles/alexander/face/curate"
    ]
    assert face_events._LAST_CURATED["alexander"] == 1000
    assert await face_events.maybe_curate_best("alexander", matches, now=2000) is False
    assert len(RecordingSession.posts) == 1, "the hourly gap must hold after unchanged"


@pytest.mark.asyncio
async def test_a_non_200_curator_response_still_fails(monkeypatch, tmp_path):
    frame = tmp_path / "face.jpg"
    frame.write_bytes(b"\xff\xd8fake")
    monkeypatch.setenv("IDENTITY_URL", "http://identity.test:18794")
    monkeypatch.setenv("IDENTITY_ADMIN_USER", "admin")
    monkeypatch.setenv("IDENTITY_ADMIN_PASSWORD", "pw")
    monkeypatch.setattr(face_events, "FACE_BEST_FRAMES_ENABLED", True)
    monkeypatch.setattr(face_events, "FACE_BEST_FRAME_MIN_GAP_S", 3600)
    face_events._LAST_CURATED.clear()
    RecordingSession.posts = []
    RecordingSession.response = CuratorResponse(status=500)
    monkeypatch.setattr(face_events.aiohttp, "ClientSession", RecordingSession)
    matches = [(0.91, str(frame), "camera_entrance1")]

    assert await face_events.maybe_curate_best("alexander", matches, now=1000) is False
    assert "alexander" not in face_events._LAST_CURATED

