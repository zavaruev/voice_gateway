"""Tests for the streamed long-sentence TTS path (10.11.2026).

coqui exposes ``/v1/audio/speech/stream``: s16le PCM at 22050 Hz, chunked as
it is synthesised, with the rate in the ``X-Sample-Rate`` header. The camera
session relays it to ``/play_audio`` resampled to 48 kHz, buffering
``_TTS_STREAM_LEAD_S`` (5.5 s) of audio BEFORE the camera POST starts, so the
speaker never drains between the ~3.6 s synthesis chunks.

The fakes below are shaped like aiohttp on purpose: a request body that is an
async generator is CONSUMED while the request is being made, so the camera
side consumes the relay generator in ``__aenter__`` — a failure raised there
is exactly what a connect refusal or a mid-stream reset looks like to the
caller, and the number of chunks recorded before it is what was already
handed to the speaker.
"""
import asyncio
import time

import pytest

import camera_client
from camera_client import (
    CameraConfig,
    CameraSession,
    TTS_PLAY_RATE,
    _TTS_STREAM_LEAD_S,
    _TTS_STREAM_MIN_CHARS,
)

from test_camera_client import _player_session

STREAM_URL = "http://tts.test:5051/v1/audio/speech/stream"
CAMERA_URL = "http://10.0.0.9/play_audio"
# > _TTS_STREAM_MIN_CHARS: this sentence must take the streamed path.
LONG = "Это длинное предложение, которое обязано идти стриминговым путём."
APOLOGY = "Простите, голос сейчас не получился."


def _lead_bytes():
    """What the relay buffers before the camera POST is allowed to start."""
    return int(_TTS_STREAM_LEAD_S * TTS_PLAY_RATE) * 2


class _Sent(str):
    """A queue item carrying speed, the way TTSentence does."""

    speed = 1.0


class _StreamResp:
    """The coqui side: status, headers, then 1-second s16le chunks @22050.

    ``stall_after`` raises asyncio.TimeoutError from ``read`` — the shape
    aiohttp's ``sock_read`` timeout takes when the synthesis stops arriving.
    """

    def __init__(self, chunks, status=200, headers=None, stall_after=None):
        self._status = status
        self._headers = (
            {"X-Sample-Rate": "22050"} if headers is None else headers
        )
        self._chunks = list(chunks)
        self._stall_after = stall_after
        self.reads = 0
        # `r.content.read(...)` — the content object IS the responder here.
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def status(self):
        return self._status

    @property
    def headers(self):
        return self._headers

    async def read(self, n):
        self.reads += 1
        if self._stall_after is not None and self.reads > self._stall_after:
            raise asyncio.TimeoutError()
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _src_chunks(n):
    """`n` seconds of source audio, one 22050*2-byte chunk per second."""
    return [bytes(b"\x01\x00") * 22050] * n


class _CameraPost:
    """The camera side of the relay.

    aiohttp streams a generator body while the request is in flight, so this
    consumes the relay in ``__aenter__`` and records every chunk it yielded.
    ``die_after=0`` is a connect refusal (nothing relayed); ``die_after=n``
    dies after n chunks were handed over (a mid-stream death).
    """

    def __init__(self, router, data):
        self._router = router
        self._data = data
        self.chunks = []

    async def __aenter__(self):
        n = 0
        die = self._router.camera_die_after
        if die is not None and die == 0:
            raise ConnectionRefusedError("camera refused the stream POST")
        async for chunk in self._data:
            if die is not None and n >= die:
                raise ConnectionResetError("camera dropped mid-stream")
            self.chunks.append(chunk)
            n += 1
        self._router.camera_posts.append(self.chunks)
        return self

    async def __aexit__(self, *exc):
        return False


class _RouterSession:
    """Fake aiohttp session: speech/stream -> coqui, anything else -> camera."""

    def __init__(self, coqui_resp, camera_die_after=None):
        self.coqui_resp = coqui_resp
        self.coqui_payloads = []
        self.camera_posts = []
        self.camera_die_after = camera_die_after

    def post(self, url, **kwargs):
        if "speech/stream" in url:
            self.coqui_payloads.append(kwargs.get("json"))
            return self.coqui_resp
        return _CameraPost(self, kwargs.get("data"))


def _stream_session():
    """A hand-built session with just enough state for the streamed path.

    Everything the stream methods touch is listed here — a missing attribute
    inside the live audio path reads as a playback fault, not as a helper gap
    (that shape has bitten this suite several times).
    """
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s.tts_stream_url = STREAM_URL
    s.tts_voice = "Zofija Kendrick"
    s.tts_api_key = ""
    s._play_audio_url = CAMERA_URL
    s._play_audio_headers = {}
    s._speaking = False
    s._speaking_until = 0.0
    s._tts_play_end = 0.0
    s._tts_playing_until = 0.0
    s._wake_suppress_until = 0.0
    s._last_play_ok = False
    s._last_tts_reply = ""
    s._last_tts_reply_at = 0.0
    # The echo ring's mechanics are tested elsewhere; here we only record that
    # every relayed chunk was registered for the echo correlator.
    s.echo_chunks = []
    s._store_tts_echo = lambda pcm, rate=TTS_PLAY_RATE: s.echo_chunks.append(
        len(pcm)
    )
    return s


# --------------------------------------------------------------------------
# the happy path: lead buffered, handoff, guards armed, echo registered
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_relays_resampled_audio_and_bails_to_nothing():
    """6 s of 22.05 kHz source -> ~6 s of 48 kHz to the camera, in ≤1 s
    pieces, with the first piece a FULL second (the lead was buffered, not
    dribbled). Nothing about the turn may claim a fallback afterwards."""
    s = _stream_session()
    resp = _StreamResp(_src_chunks(6))
    router = _RouterSession(resp)
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is True, "a healthy stream must not ask for the whole-file path"
    assert is_q is False
    assert s._stream_fallback_needed is False
    assert len(router.camera_posts) == 1
    body = router.camera_posts[0]
    total = sum(len(c) for c in body)
    assert total >= _lead_bytes(), "the camera POST started before the lead"
    assert total == pytest.approx(6 * TTS_PLAY_RATE * 2, rel=0.02)
    assert len(body[0]) == TTS_PLAY_RATE * 2, (
        "the first piece must be a full second — the lead was buffered first"
    )
    assert all(len(c) <= TTS_PLAY_RATE * 2 for c in body)
    # Our own audio is registered for the echo correlator, per chunk.
    assert sum(s.echo_chunks) == total
    assert s._last_play_ok is True
    # Every guard that keeps the room from hearing itself is armed to the END
    # of the relayed audio. Real clock, not a stub (14.10.2026: a fake `now`
    # silently tests the wrong path).
    now = time.time()
    assert s._speaking_until > now
    assert s._tts_play_end > now
    assert s._tts_playing_until > now
    assert s._wake_suppress_until > now
    assert camera_client.GLOBAL_TTS_UNTIL > now
    # The payload parity with _tts_fetch: voice from the session, no speed key
    # when the sentence is neutral.
    payload = router.coqui_payloads[0]
    assert payload["voice"] == "Zofija Kendrick"
    assert payload["input"] == LONG
    assert "speed" not in payload


@pytest.mark.asyncio
async def test_stream_carries_speed_only_when_it_differs():
    """`speed` travels exactly as in `_tts_fetch`: present at 0.9, absent at
    1.0 and at 0 (which means 'unset', not 'double speed')."""
    s = _stream_session()
    for speed, expect_key in ((0.9, True), (1.0, False), (0, False)):
        router = _RouterSession(_StreamResp(_src_chunks(6)))
        s.http_session = router
        await s._speak_stream(text=LONG, reply=LONG, speed=speed)
        payload = router.coqui_payloads[0]
        assert ("speed" in payload) is expect_key, f"speed={speed!r}"
        if expect_key:
            assert payload["speed"] == 0.9


@pytest.mark.asyncio
async def test_stream_returns_whether_the_reply_was_a_question():
    """The player opens the follow-up window from this return value, so it has
    to carry the same question detection `_speak_pcm` performs."""
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(6)))
    s.http_session = router
    is_q, ok = await s._speak_streamed(_Sent("Ты хочешь, чтобы я включила свет?"))
    assert ok is True
    assert is_q is True


@pytest.mark.asyncio
async def test_stream_ended_short_still_hands_off_what_exists():
    """A stream that ends BELOW the lead (coqui died cleanly after 3 s) hands
    the 3 s to the camera rather than saying nothing — and does NOT claim a
    fallback, because the audio already played."""
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(3)))
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is True
    assert s._stream_fallback_needed is False
    body = router.camera_posts[0]
    assert sum(len(c) for c in body) == pytest.approx(
        3 * TTS_PLAY_RATE * 2, rel=0.02
    )
    assert s._last_play_ok is True


@pytest.mark.asyncio
async def test_relay_survives_odd_sized_source_chunks():
    """`audioop.ratecv` refuses a fragment that is not a whole number of
    frames (width*nchannels = 2 bytes). Real TCP reads end at arbitrary
    offsets — measured live 11.11.2026: the first probe died with
    `error('not a whole number of frames')`, the POST bailed and the turn
    fell back to the whole-file path. The parity tail must be carried, not
    fed; the fake above yields even chunks, which is why the suite was
    green on the broken code."""
    s = _stream_session()
    chunks = [
        bytes(b"\x01\x00") * 22050 + b"\x01",  # 44101 — odd
        bytes(b"\x01\x00") * 22049 + b"\x01",  # 44099 — odd
        bytes(b"\x01\x00") * 22050,            # 44100 — even
        bytes(b"\x01\x00") * 22050,            # 44100 — even
    ]
    router = _RouterSession(_StreamResp(chunks))
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is True
    assert s._stream_fallback_needed is False
    body = router.camera_posts[0]
    total = sum(len(c) for c in body)
    # 88200 source frames = 4.0 s at 22050 -> 4.0 s at 48000: nothing lost
    # but the unrepresentable half-frame, and no double-count either.
    assert total == pytest.approx(4 * TTS_PLAY_RATE * 2, rel=0.01)
    assert s._last_play_ok is True


# --------------------------------------------------------------------------
# bails: nothing reached the speaker, the player may fall back
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_sample_rate_header_bails_before_the_camera_post():
    """Without X-Sample-Rate the relay cannot resample without pitch-shifting
    the reply — refuse BEFORE the camera hears anything, and let the player
    take the whole-file path."""
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(6), headers={}))
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is False
    assert s._stream_fallback_needed is True
    assert router.camera_posts == [], "the camera must hear nothing on a bail"
    assert s._last_play_ok is False


@pytest.mark.asyncio
async def test_non_200_bails_before_the_camera_post():
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(6), status=500))
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is False
    assert s._stream_fallback_needed is True
    assert router.camera_posts == []
    assert s._last_play_ok is False


@pytest.mark.asyncio
async def test_empty_stream_bails_instead_of_posting_silence():
    """The CUDA-tensor defect (11.11.2026) produced exactly this: HTTP 200,
    correct headers, ZERO bytes. The camera must not receive an empty POST
    and the turn must fall back."""
    s = _stream_session()
    router = _RouterSession(_StreamResp([]))
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is False
    assert s._stream_fallback_needed is True
    assert s._last_play_ok is False
    assert router.camera_posts == [[]], "an empty body must never be posted"


@pytest.mark.asyncio
async def test_camera_connect_refusal_bails_for_the_whole_file_path():
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(6)), camera_die_after=0)
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is False
    assert s._stream_fallback_needed is True
    assert router.camera_posts == []


# --------------------------------------------------------------------------
# mid-stream deaths: truncated, never re-spoken
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_stall_truncates_without_claiming_a_fallback():
    """Nothing arriving for the stall timeout ends the stream. What was
    relayed already played — a fallback WOULD speak the reply twice."""
    s = _stream_session()
    resp = _StreamResp(_src_chunks(6), stall_after=4)
    router = _RouterSession(resp)
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is True, "a truncated stream played audio; no fallback may run"
    assert s._stream_fallback_needed is False
    body = router.camera_posts[0]
    total = sum(len(c) for c in body)
    assert total < _lead_bytes(), "4 s of source is below the lead"
    assert total == pytest.approx(4 * TTS_PLAY_RATE * 2, rel=0.02)
    assert s._last_play_ok is True


@pytest.mark.asyncio
async def test_camera_dying_mid_stream_does_not_re_speak_the_reply():
    """Two chunks reached the camera, then it dropped the connection. The
    flag must stay False: the fallback path would say the whole sentence
    again over the half that already played."""
    s = _stream_session()
    router = _RouterSession(_StreamResp(_src_chunks(6)), camera_die_after=2)
    s.http_session = router

    is_q, ok = await s._speak_streamed(_Sent(LONG))

    assert ok is True
    assert s._stream_fallback_needed is False
    assert len(router.camera_posts) == 0, (
        "a POST that never completed is not recorded as delivered"
    )
    assert s._last_play_ok is True


# --------------------------------------------------------------------------
# the crossover: what streams and what does not
# --------------------------------------------------------------------------


def test_sentence_is_long_boundary():
    assert CameraSession._sentence_is_long("а" * _TTS_STREAM_MIN_CHARS) is True
    assert CameraSession._sentence_is_long("а" * (_TTS_STREAM_MIN_CHARS - 1)) is False
    assert CameraSession._sentence_is_long("") is False
    assert CameraSession._sentence_is_long(None) is False
    assert CameraSession._sentence_is_long("   ") is False


@pytest.mark.asyncio
async def test_prefetch_defers_a_long_sentence_to_the_stream_path():
    """A long sentence is returned UNFETCHED: paying whole-file synthesis
    (no first sound until done) for a sentence that will never use it would
    be the very latency this path removes."""
    s = _stream_session()
    fetched = []

    async def fake_fetch(text, speed=1.0):
        fetched.append(str(text))
        return b"pcm"

    s._tts_fetch = fake_fetch
    q = asyncio.Queue()
    await q.put(LONG)
    sent, pcm = await s._prefetch_next(q)
    assert (sent, pcm) == (LONG, None)
    assert fetched == [], "a long sentence must not be synthesised whole"

    await q.put("коротко")
    sent, pcm = await s._prefetch_next(q)
    assert pcm == b"pcm"
    assert fetched == ["коротко"]


@pytest.mark.asyncio
async def test_prefetch_fetches_everything_when_streaming_is_off():
    """Empty tts_stream_url = the old behaviour, sentence by sentence. A
    missing env var must degrade to the old latency, never to silence."""
    s = _stream_session()
    s.tts_stream_url = ""
    fetched = []

    async def fake_fetch(text, speed=1.0):
        fetched.append(str(text))
        return b"pcm"

    s._tts_fetch = fake_fetch
    q = asyncio.Queue()
    await q.put(LONG)
    sent, pcm = await s._prefetch_next(q)
    assert (sent, pcm) == (LONG, b"pcm")
    assert fetched == [LONG]


# --------------------------------------------------------------------------
# the player wiring
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_player_streams_long_sentences_and_plays_short_ones():
    """The routing in one test: the long sentence goes to `_speak_streamed`,
    the short one keeps the whole-file path, and the apology is never spent
    on a turn that spoke."""
    s = _player_session()
    s.tts_stream_url = STREAM_URL
    streamed = []

    async def fake_streamed(sent):
        streamed.append(str(sent))
        return (False, True)

    s._speak_streamed = fake_streamed
    q = asyncio.Queue()
    await q.put(_Sent(LONG))
    await q.put("коротко")
    await q.put(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert streamed == [LONG]
    assert s.played == ["коротко"]


@pytest.mark.asyncio
async def test_player_apologises_when_the_stream_never_started_and_whole_file_fails():
    """Stream bailed (nothing played), whole-file synthesis failed too — the
    turn must still attempt the apology rather than end in silence, and the
    attempt must be visible in what was asked of TTS."""
    s = _player_session()
    s.tts_stream_url = STREAM_URL
    fetched = []

    async def fake_fetch(text, speed=1.0):
        fetched.append(str(text))
        return None

    async def fake_streamed(sent):
        return (False, False)  # nothing reached the speaker

    s._tts_fetch = fake_fetch
    s._speak_streamed = fake_streamed
    q = asyncio.Queue()
    await q.put(_Sent(LONG))
    await q.put(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert fetched == [LONG, APOLOGY]


@pytest.mark.asyncio
async def test_player_does_not_fall_back_after_a_truncated_stream():
    """`streamed_ok=True` on a truncated stream must NOT trigger the
    whole-file fetch: the reply already played and a fallback speaks it
    twice (the double-speak this contract exists to prevent)."""
    s = _player_session()
    s.tts_stream_url = STREAM_URL
    fetched = []

    async def fake_fetch(text, speed=1.0):
        fetched.append(str(text))
        return b"pcm"

    async def fake_streamed(sent):
        return (False, True)  # truncated, but something played

    s._tts_fetch = fake_fetch
    s._speak_streamed = fake_streamed
    q = asyncio.Queue()
    await q.put(_Sent(LONG))
    await q.put(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert fetched == [], "no whole-file synthesis on top of a played stream"
    assert s.played == []


# --------------------------------------------------------------------------
# the env wiring
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_camera_config_carries_the_stream_url(monkeypatch):
    """main.start_camera_sessions() must pass TTS_STREAM_URL into
    CameraConfig — the same shape as the voice fill that was missed on
    10.11.2026: a CameraConfig-only test cannot see a missing fill."""
    import main as main_mod

    created = []

    class _FakeSession:
        def __init__(self, config=None, backend=None):
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
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    # VadEngine loads silero_vad.onnx from the working directory, which the
    # CI/source copy does not carry — this test is about the CONFIG wiring.
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(
        main_mod.backends, "CascadeBackend", lambda *a, **k: _StubBackend()
    )
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(main_mod, "TTS_STREAM_URL", STREAM_URL)

    await main_mod.start_camera_sessions()

    assert len(created) == 1
    cfg = created[0].config
    assert cfg.tts_stream_url == STREAM_URL
    # The dataclass default is EMPTY on purpose: empty = every sentence takes
    # the whole-file path, so a missing fill degrades to the old latency and
    # never to silence.
    assert CameraConfig(stream_name="x").tts_stream_url == ""
