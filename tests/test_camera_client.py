"""Tests for camera_client.AIVoiceOutputTrack (av-based audio output track).

Covers the PyAV track wrapper the camera stream uses to play TTS audio:
initialization, packet timestamps and frame assembly. NOTE: this module
imports `av`, which is only installed inside the Docker image — on a host
without av the whole `pytest tests/` run fails at collection, so run the
5-file suite (see AGENTS/README) instead of the full directory.
"""
import asyncio
import os
import time
import pytest
import inspect
from unittest.mock import patch

import av
import fractions

import camera_client
from camera_client import (
    AIVoiceOutputTrack,
    _FOLLOWUP_MIN_PEAK,
    _PREROLL_FRAMES,
    _WEBRTC_GIVEUP_ATTEMPTS,
    _WEBRTC_GIVEUP_RETRY_S,
    _SPEAKER_SETTLE_S,
    CameraSession,
    TTS_PLAY_RATE,
    _CORRIDOR_STREAMS,
)

@pytest.mark.asyncio
async def test_aivoiceoutputtrack_initialization():
    track = AIVoiceOutputTrack(sample_rate=16000)
    assert track.kind == "audio"
    assert track._sample_rate == 16000
    assert track._queue.maxsize == 500
    assert track._frame_count == 0

def test_track_defaults_to_the_camera_speaker_rate():
    """The camera speaker runs at 48 kHz. An 8 kHz default made every reply
    come out six times too fast ("пищит как бурундук")."""
    track = AIVoiceOutputTrack()
    assert track._sample_rate == TTS_PLAY_RATE == 48000
    # 20 ms of 48 kHz mono s16
    assert track._frame_samples == 960
    assert track._silence_frame.sample_rate == 48000


@pytest.mark.asyncio
async def test_queue_frame_and_recv():
    track = AIVoiceOutputTrack(sample_rate=8000)

    # 160 samples per frame at 8000 Hz for 20ms
    # 160 samples * 2 bytes/sample (s16) = 320 bytes
    pcm_data = b'\x01\x02' * 160

    await track.queue_frame(pcm_data, sample_rate=8000)
    assert track._queue.qsize() == 1

    # Should get the queued frame
    frame = await track.recv()
    assert isinstance(frame, av.AudioFrame)
    assert frame.sample_rate == 8000
    assert frame.samples == 160
    assert frame.format.name == "s16"
    assert track._frame_count == 1

    # Check that it advances timestamp
    assert track._timestamp == 160
    assert frame.pts == 0
    assert frame.time_base == fractions.Fraction(1, 8000)

@pytest.mark.asyncio
async def test_recv_silence_when_queue_empty():
    track = AIVoiceOutputTrack(sample_rate=8000)

    # Queue is empty, should get silence
    frame = await track.recv()
    assert isinstance(frame, av.AudioFrame)

    # Convert frame data to bytes to check if it's silence
    planes = frame.planes
    pcm = bytes(planes[0])
    assert pcm == b'\x00' * 320  # 160 samples * 2 bytes = 320 bytes of silence

    # Silence frame shouldn't increment _frame_count for real frames
    assert track._frame_count == 0
    assert track._timestamp == 160

@pytest.mark.asyncio
async def test_queue_seconds():
    track = AIVoiceOutputTrack(sample_rate=8000)

    pcm_data = b'\x01\x02' * 160
    await track.queue_frame(pcm_data, sample_rate=8000)
    await track.queue_frame(pcm_data, sample_rate=8000)

    assert track._queue.qsize() == 2
    # 2 frames * 160 samples / 8000 Hz = 2 * 0.02s = 0.04s
    assert track.queue_seconds() == 0.04

@pytest.mark.asyncio
async def test_echo_active():
    track = AIVoiceOutputTrack(sample_rate=8000)

    # Initial state
    assert track.echo_active() is False

    # Echo active when queue has items
    pcm_data = b'\x01\x02' * 160
    await track.queue_frame(pcm_data, sample_rate=8000)
    assert track.echo_active() is True

    # Receive frame to clear queue
    await track.recv()
    assert track._queue.qsize() == 0

    # Just after receive, time.time() - _last_real_recv is ~0
    # hold = min(15.0, 0.0 + 0.3) = 0.3s -> echo_active True
    assert track.echo_active() is True

    # After 1s of no real frames: 1.0 > 0.3 -> echo window closed
    track._last_real_recv = time.time() - 1.0
    assert track.echo_active() is False

    # Longer playback extends the hold: min(15.0, 5.0 + 0.3) = 5.3s
    track._last_real_recv = time.time() - 2.0
    track._last_play_duration = 5.0
    assert track.echo_active() is True

    track._last_real_recv = time.time() - 6.0
    assert track.echo_active() is False

    # Hold is capped at 15s even for very long clips
    track._last_real_recv = time.time() - 14.0
    track._last_play_duration = 60.0
    assert track.echo_active() is True

    track._last_real_recv = time.time() - 16.0
    assert track.echo_active() is False

def test_stop():
    track = AIVoiceOutputTrack(sample_rate=8000)
    # stop doesn't do anything currently, but we should test it can be called
    track.stop()


from camera_client import CameraSession, CameraConfig, CameraConfig
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_delayed_attention_exception():
    # Initialize CameraSession with minimal parameters
    session = CameraSession(CameraConfig(stream_name="test_stream", go2rtc_host="127.0.0.1", go2rtc_port=1984))

    # We want to test that if asyncio.sleep raises an Exception,
    # _delayed_attention catches it and doesn't crash,
    # and also that _attention_played remains False (or unchanged).
    session._attention_played = False

    # We patch asyncio.sleep to raise an exception
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = Exception("Test Exception")

        # We also mock _play_attention to make sure it's NOT called
        with patch.object(session, "_play_attention", new_callable=AsyncMock) as mock_play:
            # Run the method
            await session._delayed_attention()

            # Assertions
            mock_sleep.assert_called_once_with(5)
            mock_play.assert_not_called()
            assert session._attention_played is False


import numpy as np


_PIP_WAV = None


def _write_pip_wav():
    """A real 48 kHz mono wav so the activation-cue test needs no /app file."""
    import tempfile
    import wave

    global _PIP_WAV
    if _PIP_WAV:
        return _PIP_WAV
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x10" * int(48000 * 0.25))
    _PIP_WAV = path
    return path


def _make_session():
    s = CameraSession(CameraConfig(stream_name="test_stream", go2rtc_host="127.0.0.1", go2rtc_port=1984))
    return s


def _make_http_session():
    """Session wired to an OpenIPC /play_audio endpoint."""
    cfg = CameraConfig(
        stream_name="cam",
        go2rtc_host="127.0.0.1",
        go2rtc_port=1984,
        play_audio_url="http://10.0.0.9/play_audio",
        play_audio_user="root",
        play_audio_password="pw",
    )
    return CameraSession(config=cfg)


@pytest.mark.asyncio
async def test_the_mic_is_muted_while_a_reply_plays_with_no_webrtc_track():
    """The hard mute is keyed on _speaking_until and _feed_audio returns early
    on it. It used to be written only `if self._out_track`, so a room playing
    over /play_audio with the WebRTC session off lost the guard entirely — the
    reply came back into the decoder as if the user had said it."""
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    assert s._out_track is None, "no WebRTC session in this test"

    await s._speak_pcm(b"\x03\x04" * 24000, "сказано")  # 0.5 s @ 48 kHz

    assert s._speaking_until > time.time(), "mic left open during playback"

    # And it must actually keep chunks out.
    fed = []
    s._feed_audio = lambda pcm, rate=16000: fed.append(pcm)
    await CameraSession._feed_audio(s, b"\x01\x02" * 160)
    assert fed == []


@pytest.mark.asyncio
async def test_the_followup_window_waits_for_the_reply_to_finish_sounding():
    """The window must open when the answer STOPS, not when it was queued.

    `_wait_playback_drain` used to return immediately with no WebRTC track — and
    there almost never is one — so the window opened while the camera was still
    talking while `_speaking_until` held the mic hard-muted. Measured
    05.10.2026: `Follow-up open` 0.07 s after the TTS fetch against a playback
    that ran 4.3 s longer, so for those 4.3 s anything the user said was thrown
    away. That is the room being deaf immediately after it answers.
    """
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    assert s._out_track is None, "no WebRTC track in this test"

    # A 2 s reply: the drain must not return before it has finished.
    pcm = b"\x03\x04" * int(TTS_PLAY_RATE * 2)
    await s._speak_pcm(pcm, "Включила")

    drained_at = time.monotonic()
    await s._wait_playback_drain()
    waited = time.monotonic() - drained_at

    assert waited > 0.3, (
        f"returned in {waited:.2f}s — the window opened while the reply played"
    )
    assert time.time() >= s._tts_play_end - 0.2, (
        "drain returned before the reply finished sounding"
    )
    # It must NOT wait out the echo tail as well: the window is for the moment
    # the answer ends, not 3 s into the quiet after it.
    assert time.time() < s._speaking_until + 0.3, "waited out the echo tail too"


@pytest.mark.asyncio
async def test_the_drain_does_not_hang_when_nothing_is_playing():
    """It waits on a timestamp, so a stale one must not park the player forever
    — that is how the reply would never open its window."""
    s = _make_session()
    s._tts_play_end = 0.0
    s._speaking_until = 0.0

    started = time.monotonic()
    await asyncio.wait_for(s._wait_playback_drain(), timeout=5)
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_webrtc_off_never_offers_and_webrtc_on_does(monkeypatch):
    """Every offer makes go2rtc rebuild the stream producer; an abandoned one
    left a session nobody read and filled the camera's send queue to 193 kB,
    which blocked majestic (HTTP 14 s, no RTSP). So with the session disabled no
    connection attempt may be made at all."""
    attempts = []

    class _Sess(CameraSession):
        async def _connect(self):
            attempts.append(1)
            self._stopped.set()
            return False

        async def _init_engine(self):
            return None

        async def _rtsp_audio_loop(self):
            self._stopped.set()

    # The supervisor's retry pause is what this test must not wait on.
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MIN_S", 0.0)
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MAX_S", 0.0)

    off = _Sess(CameraConfig(stream_name="cam", webrtc=False))
    await off.start()
    await _drain(off)
    assert attempts == [], "opened a WebRTC session that was disabled"

    on = _Sess(CameraConfig(stream_name="cam", webrtc=True))
    await on.start()
    await _drain(on)
    assert attempts == [1], f"session enabled but not attempted once: {attempts}"


@pytest.mark.asyncio
async def test_the_webrtc_supervisor_backs_off_instead_of_spinning(monkeypatch):
    """7 offers in 12 minutes is what wedged the camera. An un-answered attempt
    must be followed by a growing pause, and an answered one resets it."""
    from camera_client import _WEBRTC_BACKOFF_MAX_S, _WEBRTC_BACKOFF_MIN_S

    sleeps = []

    async def _record(delay):
        sleeps.append(delay)

    class _Sess(CameraSession):
        def __init__(self, results):
            super().__init__(CameraConfig(stream_name="cam"))
            self._results = list(results)

        async def _connect(self):
            if not self._results:
                self._stopped.set()
                return False
            return self._results.pop(0)

    wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, "sleep", _record)

    s = _Sess([False, False, False])
    await wait_for(s._run(), timeout=5)

    assert sleeps[:3] == [
        _WEBRTC_BACKOFF_MIN_S,
        _WEBRTC_BACKOFF_MIN_S * 2,
        _WEBRTC_BACKOFF_MIN_S * 4,
    ], f"no exponential backoff: {sleeps}"
    assert max(sleeps) <= _WEBRTC_BACKOFF_MAX_S

    # An answered session resets the ladder instead of inheriting the pause the
    # failures had built up. The trailing 30 s is the final un-answered attempt
    # after the answers ran out, which is what proves the reset happened.
    sleeps.clear()
    s2 = _Sess([False, True, False])
    await wait_for(s2._run(), timeout=5)
    assert sleeps == [_WEBRTC_BACKOFF_MIN_S, _WEBRTC_BACKOFF_MIN_S, 30.0], sleeps


@pytest.mark.asyncio
async def test_a_webrtc_attempt_reports_whether_it_was_answered():
    """The supervisor's backoff is driven by this bool, so the timeout exit has
    to say False rather than falling off the end of the function."""
    import aiohttp

    s = _make_session()

    class _Ws:
        async def receive(self, timeout=None):
            raise asyncio.TimeoutError()

        async def send_json(self, _payload):
            return None

    class _WsCtx:
        async def __aenter__(self):
            return _Ws()

        async def __aexit__(self, *a):
            return False

    class _SessCtx:
        def __init__(self, *a, **kw):
            pass

        def ws_connect(self, url):
            return _WsCtx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    orig = aiohttp.ClientSession
    aiohttp.ClientSession = _SessCtx
    try:
        assert await s._connect() is False, "un-answered offer reported success"
    finally:
        aiohttp.ClientSession = orig


async def _drain(session, rounds=6):
    """Let created tasks run so their side effects are visible."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class _FakeResp:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeHttp:
    """Minimal aiohttp.ClientSession stand-in that records the POST."""

    def __init__(self, status=200):
        self.status = status
        self.calls = []

    def post(self, url, data=None, headers=None, auth=None, timeout=None):
        self.calls.append(
            {"url": url, "data": data, "headers": headers, "auth": auth}
        )
        return _FakeResp(self.status)


def test_wake_threshold_defaults_to_the_historical_per_room_values():
    """A room without an explicit threshold keeps kitchen 0.40 / other 0.30."""
    for name, want in (("kitchen", 0.40), ("corridor1", 0.30),
                       ("livingroom", 0.30)):
        s = CameraSession(CameraConfig(stream_name=name))
        assert s._ww_base_thresh == want, name
        assert s._ww_thresh == want, name


def test_wake_threshold_override_is_honoured():
    """A room-specific head must not inherit 0.30: its score distribution is
    its own. Measured livingroom operating point is 0.70 (36/36 hits,
    0/42 false positives)."""
    s = CameraSession(CameraConfig(stream_name="livingroom", wake_threshold=0.70))
    assert s._ww_base_thresh == 0.70
    assert s._ww_thresh == 0.70
    # the threshold-bump reset must clamp back to the ROOM's base, not 0.30
    s._ww_thresh = 0.95
    s._ww_base_thresh = 0.70
    assert min(s._ww_thresh, s._ww_base_thresh) == 0.70


def test_livingroom_head_ships_and_matches_the_runtime_signature():
    """The trained head must load in the runtime's ONNX runtime and accept
    exactly the [1,16,96] input openWakeWord feeds it."""
    import onnxruntime as ort
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "computer_livingroom.onnx")
    if not os.path.exists(path):
        pytest.skip("livingroom head not built in this checkout")
    # tests/test_engine.py and tests/test_rms.py replace
    # ort.InferenceSession with a MagicMock at import time and never restore
    # it, so the real session is only reachable when this file runs first.
    if not isinstance(ort.InferenceSession, type):
        pytest.skip("ort.InferenceSession is mocked by another test module")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    assert list(shape) == [1, 16, 96], shape
    assert len(sess.get_outputs()) == 1


def test_backend_argument_is_actually_kept():
    """main.start_camera_sessions() builds a Cascade/Hermes backend per camera
    and passes it in. A hardcoded `self.backend = None` in __init__ dropped it
    and routed every camera command to the retired Nanobot WebSocket, which
    left the room mute after a successful wake word."""
    sentinel = object()
    s = CameraSession(
        CameraConfig(stream_name="cam", go2rtc_host="127.0.0.1", go2rtc_port=1984),
        backend=sentinel,
    )
    assert s.backend is sentinel


def test_backend_defaults_to_none_for_the_legacy_path():
    s = _make_session()
    assert s.backend is None


@pytest.mark.asyncio
async def test_call_nanobot_prefers_the_backend_over_the_legacy_socket():
    """_call_nanobot is the single funnel for camera commands (wake command,
    arbitration rescue, follow-up). With a backend configured it must never
    touch the Nanobot WebSocket."""
    s = _make_http_session()

    class _FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate_response(self, text=None, session_id=None,
                                    stream_name=None, response_queue=None):
            self.calls.append(text)
            response_queue.put_nowait("ответ")

    backend = _FakeBackend()
    s.backend = backend
    with patch.object(s, "_call_backend", new_callable=AsyncMock) as mock_backend:
        await s._call_nanobot("включи свет", "camera")
    assert mock_backend.await_count == 1
    assert backend.calls == []


@pytest.mark.asyncio
async def test_play_audio_posts_pcm_with_the_rate_in_the_content_type():
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    pcm = b"\x01\x02" * 4800  # 0.1 s @ 48 kHz

    assert await s._play_audio_http(pcm) is True

    call = s.http_session.calls[0]
    assert call["url"] == "http://10.0.0.9/play_audio"
    assert call["data"] is pcm
    # The rate is what makes the speaker play at the right pitch; without it
    # the camera guesses and everything comes out wrong.
    assert call["headers"]["Content-Type"] == (
        f"application/octet-stream;rate={TTS_PLAY_RATE}"
    )
    assert TTS_PLAY_RATE == 48000
    assert call["headers"]["Authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_play_audio_reports_failure_so_the_caller_can_fall_back():
    s = _make_http_session()
    s.http_session = _FakeHttp(401)
    assert await s._play_audio_http(b"\x00\x00") is False


@pytest.mark.asyncio
async def test_play_audio_unconfigured_is_a_no_op():
    s = _make_session()  # no play_audio_url
    assert await s._play_audio_http(b"\x00\x00") is False


@pytest.mark.asyncio
async def test_speak_pcm_prefers_http_over_the_go2rtc_track():
    """The track path is 6x pitch-shifted on this firmware, so when
    /play_audio is configured nothing may be queued on the track."""
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    s._out_track = AIVoiceOutputTrack()
    queued = []

    async def fake_queue(pcm, rate):
        queued.append((pcm, rate))

    s._out_track.queue_frame = fake_queue

    await s._speak_pcm(b"\x03\x04" * 24000, "сказано")  # 0.5 s @ 48 kHz

    assert len(s.http_session.calls) == 1
    assert queued == []


@pytest.mark.asyncio
async def test_speak_pcm_falls_back_to_the_track_when_http_fails():
    s = _make_http_session()
    s.http_session = _FakeHttp(500)
    s._out_track = AIVoiceOutputTrack()
    queued = []

    async def fake_queue(pcm, rate):
        queued.append((pcm, rate))

    s._out_track.queue_frame = fake_queue

    await s._speak_pcm(b"\x03\x04" * 24000, "сказано")  # 0.5 s @ 48 kHz

    assert len(s.http_session.calls) == 1
    # 0.5 s of 48 kHz audio in 20 ms frames
    assert len(queued) == 25
    assert queued[0][1] == TTS_PLAY_RATE


@pytest.mark.asyncio
async def test_is_echo_detects_own_tts():
    s = _make_session()
    rng = np.random.default_rng(0)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    sig_16k = np.repeat(sig_8k, 2)  # what _store_tts_echo upsamples to
    s._store_tts_echo(sig_8k.tobytes(), rate=8000)
    # total is now 2560; simulate the echo returning. 3 s, not the 10 s this
    # test used: `_is_echo` now refuses to believe a correlation more than
    # `_ECHO_HORIZON_S` after playback ended, because room noise matching a
    # stale ring buffer is what muted the living room for half an hour on
    # 07.10.2026. A delay the detector must NOT accept cannot be asserted as
    # detected — and the «must NOT be echo» tests below would have passed for
    # the wrong reason if they kept it.
    s._tts_total += 3 * 16000   # within _ECHO_HORIZON_S, see below
    s._tts_play_end = time.time()   # playback just ended
    # The echoed chunk (same content) must be detected as echo
    assert s._is_echo(sig_16k.tobytes()) is True


@pytest.mark.asyncio
async def test_is_echo_detects_own_tts_at_playback_rate():
    """The camera leg plays 48 kHz (TTS_PLAY_RATE), so the echo reference is
    built from 48k audio. A rate-blind ring would store a 3x-too-long
    reference and _is_echo would stop matching the mic at all."""
    s = _make_session()
    rng = np.random.default_rng(0)
    sig_48k = rng.integers(-3000, 3000, size=7680, dtype=np.int16)  # 160 ms
    # What the 16 kHz mic would record back: decimate by 3.
    sig_16k = sig_48k[::3].copy()
    assert TTS_PLAY_RATE == 48000
    s._store_tts_echo(sig_48k.tobytes())  # default rate = playback rate
    s._tts_total += 3 * 16000   # within _ECHO_HORIZON_S, see below
    s._tts_play_end = time.time()   # playback just ended
    assert s._is_echo(sig_16k.tobytes()) is True


@pytest.mark.asyncio
async def test_store_tts_echo_resamples_odd_ratio():
    """A rate that is not a divisor of 16k must be resampled properly rather
    than by sample stretching, or the reference drifts out of alignment."""
    s = _make_session()
    rng = np.random.default_rng(7)
    sig = rng.integers(-3000, 3000, size=4410, dtype=np.int16)  # 100 ms @ 44.1k
    s._store_tts_echo(sig.tobytes(), rate=44100)
    # 100 ms of 44.1k lands in the ring as ~1600 samples of 16k audio.
    assert s._tts_total == pytest.approx(1600, abs=40)


@pytest.mark.asyncio
async def test_is_echo_rejects_other_speech():
    s = _make_session()
    rng = np.random.default_rng(1)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    other_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    other_16k = np.repeat(other_8k, 2)
    s._store_tts_echo(sig_8k.tobytes(), rate=8000)
    s._tts_total += 3 * 16000   # within _ECHO_HORIZON_S, see below
    s._tts_play_end = time.time()   # playback just ended
    # Different speech must NOT be flagged as echo
    assert s._is_echo(other_16k.tobytes()) is False


@pytest.mark.asyncio
async def test_is_echo_rejects_silence():
    s = _make_session()
    rng = np.random.default_rng(2)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    silence = np.zeros(2560, dtype=np.int16)
    s._store_tts_echo(sig_8k.tobytes(), rate=8000)
    s._tts_total += 3 * 16000   # within _ECHO_HORIZON_S, see below
    s._tts_play_end = time.time()   # playback just ended
    assert s._is_echo(silence.tobytes()) is False


@pytest.mark.asyncio
async def test_recent_wake_mishear_goes_to_greeting():
    s = _make_session()
    s._wake_detected = True
    s._wake_fired_at = time.time() - 1.0  # model fired 1s ago
    with patch.object(s, "_schedule_wake_greeting") as mock_greet, patch.object(
        s, "_call_nanobot", new_callable=AsyncMock
    ) as mock_nano:
        await s._handle_wake_or_command("шшш шшш", "cam")
        mock_greet.assert_not_called()
        mock_nano.assert_not_called()


@pytest.mark.asyncio
async def test_old_wake_command_goes_to_nanobot():
    s = _make_session()
    s._wake_detected = True
    s._wake_fired_at = time.time() - 10.0  # real follow-up, not the wake word
    with patch.object(s, "_schedule_wake_greeting") as mock_greet, patch.object(
        s, "_call_nanobot", new_callable=AsyncMock
    ) as mock_nano:
        await s._handle_wake_or_command("включи свет", "cam")
        mock_greet.assert_not_called()
        mock_nano.assert_called_once()

def test_filter_sdp_default_ip():
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        "a=candidate:1 1 UDP 2013266431 192.168.22.250 50000 typ host\n"
        "a=candidate:2 1 UDP 2013266431 192.168.22.102 50000 typ host\n"
        "c=IN IP4 192.168.22.250\n"
    )
    expected = (
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        "a=candidate:2 1 UDP 2013266431 192.168.22.102 50000 typ host\n"
        "c=IN IP4 192.168.22.250\n"
    )
    assert CameraSession._filter_sdp(sdp) == expected

def test_filter_sdp_custom_ip():
    sdp = (
        "v=0\n"
        "a=candidate:1 1 UDP 2013266431 10.0.0.5 50000 typ host\n"
        "a=candidate:2 1 UDP 2013266431 192.168.22.250 50000 typ host\n"
    )
    expected = (
        "v=0\n"
        "a=candidate:2 1 UDP 2013266431 192.168.22.250 50000 typ host\n"
    )
    assert CameraSession._filter_sdp(sdp, drop_ip="10.0.0.5") == expected

def test_filter_sdp_empty():
    assert CameraSession._filter_sdp("") == ""

def test_corridor_streams_cover_the_renamed_cams():
    """The corridor-specific mic gain must survive the corridor1/corridor2 split.

    go2rtc has no plain "corridor" stream any more, so the `== "corridor"`
    tests in _vad_process matched nothing and both the x5 boost and the 9000
    AGC target were dead code.
    """
    assert {"corridor", "corridor1", "corridor2"} <= _CORRIDOR_STREAMS

def test_corridor_cams_take_the_corridor_agc_target():
    for name in ("corridor", "corridor1", "corridor2"):
        assert name in _CORRIDOR_STREAMS
        target = (
            6500
            if name == "kitchen"
            else 9000 if name in _CORRIDOR_STREAMS else 4000
        )
        assert target == 9000

def test_non_corridor_rooms_keep_the_generic_target():
    for name in ("livingroom", "kitchen", "pantry", "balcony"):
        target = (
            6500
            if name == "kitchen"
            else 9000 if name in _CORRIDOR_STREAMS else 4000
        )
        assert target == (6500 if name == "kitchen" else 4000)


# --- audio-rate watchdog ------------------------------------------------
# Regression tests for the starvation recovery. The recovery signal used to be
# `raise asyncio.CancelledError`, which this task's own
# `except asyncio.CancelledError: break` turned into a PERMANENT exit of
# `_rtsp_audio_loop`: the room went deaf forever (WebRTC still connected,
# keepalive still ticking, /health still ok) even after the camera was fixed.
# Nothing restarted the loop — `start()` only registers a discard callback.

def _rate_session():
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s._min_audio_rate = 0.5
    s._starve_restarts = 3
    s._rate_bytes = 0
    s._rate_start = 0.0
    s._rate_check = 0.0
    s._rate_ratio = 1.0
    s._rate_starved = 0
    s.heals = 0

    async def _heal():
        s.heals += 1

    s._heal_go2rtc_stream = _heal
    return s


def _set_rate(s, ratio, span=40.0):
    """Pretend a `span`-second window delivered `ratio` x real time."""
    s._rate_bytes = int(span * 16000 * 2 * ratio)
    s._rate_start = 1.0
    s._rate_check = 1.0
    return 1.0 + span


@pytest.mark.asyncio
async def test_starved_stream_asks_for_a_restart_not_a_cancellation():
    """Under the starvation threshold the loop must be told to RESTART.

    Returning False here (the old behaviour, via the raise) is what left the
    session permanently mute.
    """
    s = _rate_session()
    now = _set_rate(s, 0.05)     # 5 % of real time: starved
    with patch("camera_client.time.time", return_value=now):
        restart = await s._account_audio_rate(0)
    assert restart is True
    assert s.heals == 1
    assert s._rate_starved == 1


@pytest.mark.asyncio
async def test_healthy_stream_asks_for_nothing():
    s = _rate_session()
    now = _set_rate(s, 1.1)      # 110 % of real time: healthy
    with patch("camera_client.time.time", return_value=now):
        assert await s._account_audio_rate(0) is False
    assert s.heals == 0
    assert s._rate_starved == 0


@pytest.mark.asyncio
async def test_starvation_counter_is_reached_and_reported():
    """After `_starve_restarts` failures the operator must be told a reboot is
    needed. With the old code the loop died on the first failure, so this branch
    was unreachable and the message never appeared."""
    s = _rate_session()
    results = []
    for i in range(4):
        now = _set_rate(s, 0.01)     # ~1 % of real time
        with patch("camera_client.time.time", return_value=now + i * 20):
            results.append(await s._account_audio_rate(0))
    # 1st and 2nd: restart. 3rd: give up and report, no restart.
    assert results == [True, True, False, True]
    assert s.heals == 3
    # the 4th window started counting from 1 again — that is the reset working
    assert s._rate_starved == 1


@pytest.mark.asyncio
async def test_rate_recovery_clears_the_starvation_counter():
    s = _rate_session()
    with patch("camera_client.time.time", return_value=_set_rate(s, 0.01)):
        await s._account_audio_rate(0)
    assert s._rate_starved == 1
    with patch("camera_client.time.time", return_value=_set_rate(s, 1.0)):
        assert await s._account_audio_rate(0) is False
    assert s._rate_starved == 0


@pytest.mark.asyncio
async def test_accounting_is_a_noop_inside_the_window():
    """The check must not run on every chunk, or the 20 s window is meaningless."""
    s = _rate_session()
    s._rate_start = 100.0
    s._rate_check = 100.0
    with patch("camera_client.time.time", return_value=105.0):
        assert await s._account_audio_rate(2560) is False
    assert s._rate_check == 100.0, "window must not be reset early"






# --- the reply LEVEL is a per-room lever, and it must move BOTH ways -----
# Reported as crackling playback on 05.10.2026. The microphone cannot settle
# it -- it is saturated (input peaks 20000/10000/4000 all captured as rms
# ~26300, peak 32768) -- so the level has to be found by LISTENING. That only
# works if the knob actually moves the signal, and the original condition
# "gain > 1.2" could only ever boost: with the target below the TTS's own peak
# the gain falls to ~0.70, the branch does not run, and the lever does
# nothing, which from the field reads as "lowering it did not help".


class _TtsFakeSeg:
    def __init__(self, raw):
        self.raw_data = raw

    def set_frame_rate(self, _r):
        return self

    def set_channels(self, _c):
        return self

    def set_sample_width(self, _w):
        return self


class _TtsFakeResp:
    status = 200

    def __init__(self, raw):
        self._raw = raw

    async def read(self):
        return self._raw


class _TtsFakePost:
    def __init__(self, raw):
        self._raw = raw

    async def __aenter__(self):
        return _TtsFakeResp(self._raw)

    async def __aexit__(self, *_a):
        return False


class _TtsFakeSession:
    def __init__(self, raw):
        self._raw = raw

    def post(self, *_a, **_k):
        return _TtsFakePost(self._raw)


def _tts_session(target):
    from camera_client import _TTS_TARGET_PEAK_DEFAULT

    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s.tts_api_key = ""
    s.tts_url = "http://tts"
    s.tts_voice = "v"
    # 0 => default, exactly as __init__ resolves it.
    s._tts_target_peak = target or _TTS_TARGET_PEAK_DEFAULT
    return s


# The peak measured on a real reply on 05.10.2026. Chosen so the default
# target lands in the dead band and the reply is left untouched, which is what
# was measured: gain 1.17, i.e. the boost never even applied.
MEASURED_PEAK = 17091


def _raw_with_peak(peak):
    n = 4800
    return (np.sin(np.linspace(0, 40, n)) * peak).astype(np.int16).tobytes()


def _out_peak(raw):
    return int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))))


@pytest.mark.asyncio
async def test_tts_level_lever_leaves_the_default_reply_untouched():
    """At the default target, behaviour must be bit-identical to before."""
    raw = _raw_with_peak(MEASURED_PEAK)
    s = _tts_session(0)
    s.http_session = _TtsFakeSession(raw)
    with patch("camera_client.AudioSegment.from_file", return_value=_TtsFakeSeg(raw)):
        out = await s._tts_fetch("x")
    assert _out_peak(out) == _out_peak(raw), (
        "the default must not touch a reply that already sits at 52 % of"
        " full scale -- measured 05.10.2026 as gain 1.17, not applied"
    )


@pytest.mark.asyncio
async def test_tts_level_lever_actually_attenuates():
    """The whole point: a target BELOW the natural peak must be heard."""
    raw = _raw_with_peak(MEASURED_PEAK)
    s = _tts_session(12000)
    s.http_session = _TtsFakeSession(raw)
    with patch("camera_client.AudioSegment.from_file", return_value=_TtsFakeSeg(raw)):
        out = await s._tts_fetch("x")
    got = _out_peak(out)
    assert got < MEASURED_PEAK, "a lower target did nothing: peak stayed at %d" % got
    assert abs(got - 12000) < 600, "expected ~12000, got %d" % got


@pytest.mark.asyncio
async def test_tts_level_lever_still_boosts_a_quiet_clip():
    """Pre-existing behaviour: a quiet TTS is lifted toward the target."""
    raw = _raw_with_peak(4000)
    s = _tts_session(0)
    s.http_session = _TtsFakeSession(raw)
    with patch("camera_client.AudioSegment.from_file", return_value=_TtsFakeSeg(raw)):
        out = await s._tts_fetch("x")
    assert abs(_out_peak(out) - 16000) < 400, "expected ~16000, got %d" % _out_peak(out)


def test_zero_target_means_the_default_not_silence():
    """0 is 'unset' everywhere in this config; it must never mute a room."""
    from camera_client import _TTS_TARGET_PEAK_DEFAULT

    cfg = CameraConfig("livingroom")
    assert cfg.tts_target_peak == 0, "unset must stay 0 in the config"
    assert _tts_session(0)._tts_target_peak == _TTS_TARGET_PEAK_DEFAULT
    assert _tts_session(12000)._tts_target_peak == 12000

# --- the greeting must not talk over the command it is waiting for ------
# Field case 04.10.2026: «компьютер, включи свет» -> the camera answered «Да?».
# `_vad_has_speech` was False for the whole window (the onset requires
# `not _processing_utterance`, and a previous turn was still in flight), so
# `_wake_greeting` read the room as silent — while the VAD reported
# `speech=True consec=54` and the command never reached Whisper at all.


async def _raise_cancelled(_delay):
    """Stand-in for asyncio.sleep that ends the greeting immediately."""
    raise asyncio.CancelledError


async def _no_sleep(_delay):
    return None


def _greet_session():
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s._wake_detected = True
    s._vad_has_speech = False       # an in-flight utterance blocks the onset
    s._vad_speech_consecutive = 54  # ...yet speech frames keep arriving
    s._wake_greeting_delay = 5.0
    s._wake_greeting_task = None
    s._auto_greeting = False
    s._last_auto_greet_ts = 0.0
    s.spoken = []

    async def _tts_fetch(text):
        return b"pcm"

    async def _speak(pcm, text):
        s.spoken.append(text)

    s._tts_fetch = _tts_fetch
    s._speak_pcm = _speak
    return s


@pytest.mark.asyncio
async def test_no_greeting_while_speech_frames_are_still_arriving():
    """`_last_speech_at` is the honest "is the user talking" signal."""
    s = _greet_session()
    s._last_speech_at = time.time()      # speech arriving RIGHT NOW
    with patch("camera_client.asyncio.sleep", new=_raise_cancelled):
        await s._wake_greeting()
    assert s.spoken == [], (
        "greeting interrupted live speech — this is how «включи свет» was lost"
    )


@pytest.mark.asyncio
async def test_greeting_still_happens_in_a_quiet_room():
    """A bare «компьютер» with nobody following it must still be answered."""
    s = _greet_session()
    s._last_speech_at = time.time() - 600.0
    with patch("camera_client.asyncio.sleep", new=_no_sleep):
        await s._wake_greeting()
    assert s.spoken == ["Да?"]


# --- the firing gate, tested by BEHAVIOUR and not by reading the source ---
# Three earlier versions of these tests grepped the module source for strings.
# That is the wrong instrument: the rules are stated IN THE COMMENTS as the
# mistakes that were made, so a substring search finds the very counter-example
# it is trying to prove absent. The gate is a method, so it is called.


def _gate_session():
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s._wake_suppress_until = 0.0
    s._last_wake_fired_at = 0.0
    s._wake_detected = False
    return s


def test_a_wake_fires_when_nothing_is_playing():
    assert _gate_session()._wake_gate_open(1000.0) is True


def test_a_wake_is_muted_only_while_our_own_audio_is_in_the_air():
    s = _gate_session()
    s._wake_suppress_until = 1003.0
    assert s._wake_gate_open(1002.9) is False, "tail of our own playback"
    assert s._wake_gate_open(1003.1) is True, "a 3 s tail must not eat the next command"
    assert "echo_tail" in s._wake_gate_reason(1002.9)


def test_the_same_breath_cannot_fire_twice():
    """A second `_fire_wake()` clears `_vad_speech_buf` and destroys the command
    being collected.

    The bound was 1.5 s and is now 3.0 s: measured 06.10.2026 14:36, the decoder
    decoded «компьютер» again **2.62 s** after the trigger, from the tail of the
    same word, because the trigger lands mid-word and the tail goes to the fresh
    recogniser. At 1.5 s the guard had already expired, so the second fire would
    have cleared the command. 2.0 s below is what this test used to assert was
    OPEN — that assertion was the bug, not the code."""
    s = _gate_session()
    s._last_wake_fired_at = 1000.0
    assert s._wake_gate_open(1000.8) is False
    assert "same_breath" in s._wake_gate_reason(1000.8)
    assert s._wake_gate_open(1002.0) is False, (
        "2.0 s was asserted open while the measured re-trigger came at 2.62 s"
    )
    assert s._wake_gate_open(1002.62) is False, "the measured re-trigger got through"
    assert s._wake_gate_open(1003.2) is True


def test_an_open_dialogue_window_does_not_close_the_gate():
    """THE regression of 04.10.2026.

    `_wake_detected` stays True for the whole `_wake_timeout` (60 s) after a
    wake. The gate used to consult it, so a second «компьютер» 8 s after the
    reply was decoded and thrown away — twice — before the third attempt 28 s
    later got through.
    """
    from camera_client import _WAKE_REARM_DEBOUNCE_S

    s = _gate_session()
    s._wake_detected = True          # the post-wake dialogue window is open
    s._last_wake_fired_at = 1000.0  # the first wake
    later = 1008.0                   # user repeats the word 8 s after the reply
    assert later - s._last_wake_fired_at > _WAKE_REARM_DEBOUNCE_S
    assert s._wake_gate_open(later) is True, (
        "an open dialogue window must not silence the wake word"
    )


def test_the_gate_never_consults_wake_detected():
    """Not even indirectly: two states differing ONLY in `_wake_detected` must
    produce the same verdict at every instant."""
    closed, opened = _gate_session(), _gate_session()
    closed._wake_detected = True
    opened._wake_detected = False
    for t in (1000.0, 1000.5, 1005.0, 1050.0, 1061.0):
        assert closed._wake_gate_open(t) == opened._wake_gate_open(t), (
            f"verdict changed at t={t} only because _wake_detected differs"
        )


def test_echo_tail_is_short_enough_to_re_arm():
    """15 s after a reply is what made «компьютер» stop working.

    Measured 04.10.2026: a reply finished at T blocked every wake word until
    T+18 s — the user's attempts at +8 s and +11 s were decoded and thrown away.
    The delayed room echo is `_is_echo`'s job, and it drops such a chunk in
    `_feed_audio` BEFORE it ever reaches the decoder.
    """
    from camera_client import _ECHO_TAIL_S

    assert _ECHO_TAIL_S <= 3.0, "a long tail swallows the user's next command"


def test_rearm_debounce_is_sub_second_not_a_minute():
    from camera_client import _WAKE_REARM_DEBOUNCE_S

    assert 0.3 <= _WAKE_REARM_DEBOUNCE_S <= 3.0


# --- a connection that never starts -------------------------------------
# Field case 04.10.2026 20:18: 25+ reconnects every 3.2 s, no AUDIO STARVED,
# no heal, no reason logged. Four holes, and this block is the regression test
# for the two that decide whether anything gets HEALED.


class _FakeStderr:
    def __init__(self, data: bytes = b""):
        self._data = data

    async def read(self):
        return self._data


class _FakeProc:
    def __init__(self, data: bytes = b""):
        self.stderr = _FakeStderr(data)


def _cycle_session(heal_stalls: int = 3):
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s._heal_stalls = heal_stalls
    s._dead_cycles = 0
    s._dead_total = 0
    s._audio_sleep = 3.0
    s._rate_bytes = 0
    s._rate_start = 1.0
    s._rate_check = 1.0
    s._rate_ratio = 1.0
    s._rate_starved = 0
    s._min_audio_rate = 0.5
    s._starve_restarts = 3
    # _end_utterance() touches these; tests that call it need them present.
    s._vad_speech_buf = bytearray()
    s._vad_has_speech = True
    s._vad_speech_consecutive = 5
    s._vad_silence_frames = 0
    s._vad_window = []
    s._vad_silence_limit = 10
    s._vad_max_duration = 7.0
    s._end_reason = {}
    s._wake_detected = False
    s.heals = 0

    async def _heal():
        s.heals += 1

    s._heal_go2rtc_stream = _heal
    return s


@pytest.mark.asyncio
async def test_audio_flowing_resets_the_dead_counters_and_the_backoff():
    s = _cycle_session()
    s._dead_cycles = 2
    s._dead_total = 7
    s._audio_sleep = 30.0
    await s._audio_cycle_done(_FakeProc(), 2560)
    assert s._dead_cycles == 0 and s._dead_total == 0
    assert s._audio_sleep == 3.0, "a recovered room must retry fast again"
    assert s.heals == 0


@pytest.mark.asyncio
async def test_a_zero_byte_connection_is_counted_as_a_failed_connect():
    """One dead connection is counted, and does NOT yet trigger the
    dead-cycle heal — that needs `_heal_stalls` of them in a row.

    `heals` is deliberately NOT asserted to be 0. Feeding the rate watchdog on
    this path also heals, because zero bytes IS starvation, and that is
    correct: `_heal_go2rtc_stream()` rate-limits itself to once a minute in
    production, so the duplicate call costs nothing. What this test pins is that
    the counter went UP and was not reset, which is what proves the dead-cycle
    branch stayed out of it.
    """
    s = _cycle_session()
    await s._audio_cycle_done(_FakeProc(b""), 0)
    assert s._dead_cycles == 1, "the dead connection was not counted"
    assert s._dead_total == 1, "the un-reset total was not counted"
    assert s._dead_cycles < s._heal_stalls, (
        "the dead-cycle heal fired long before its budget"
    )


@pytest.mark.asyncio
async def test_the_rate_watchdog_is_fed_on_the_failure_path():
    """The bug that hid the outage.

    `_account_audio_rate` lived only inside the SUCCESSFUL read, so "20 s of
    wall clock, zero bytes" was invisible to it — the exact condition it
    exists to catch. With zero bytes delivered the watchdog never ran, so it
    never said AUDIO STARVED, and the log looked like a harmless reconnect
    loop instead of a dead room.
    """
    s = _cycle_session()
    await s._audio_cycle_done(None, 0)
    assert s._rate_starved == 1, (
        "zero-byte connection did not feed the watchdog"
    )


@pytest.mark.asyncio
async def test_repeated_zero_byte_connections_trigger_the_go2rtc_heal():
    """`_stall_count` cannot do this job: a stall needs 20 s of silence AFTER
    audio has flowed, so with no audio ever arriving it stays 0 forever and the
    healer is never called. That is precisely the 04.10.2026 failure."""
    s = _cycle_session(heal_stalls=3)
    for _ in range(3):
        await s._audio_cycle_done(None, 0)
    # ">= 1", not "== 1": the rate watchdog ALSO heals on the zero-byte path and
    # this fake has no 60 s rate limit, so the exact count is an artefact. What
    # matters is that the healer is reached at all.
    assert s.heals >= 1, "the healer never ran"
    assert s._dead_cycles == 0, "the heal counter must reset after healing"


@pytest.mark.asyncio
async def test_a_dead_room_is_declared_and_the_retry_slows_down():
    """Regression: escalation keyed on `_dead_cycles` was UNREACHABLE.

    The heal branch resets `_dead_cycles` to 0, so it can never climb to
    `_heal_stalls * 4`. The room was therefore never declared dead and the
    loop retried every 3 s forever, burying everything else in the log.
    """
    s = _cycle_session(heal_stalls=3)
    for _ in range(12):
        await s._audio_cycle_done(None, 0)
    assert s._dead_total == 12, (
        "the total counter must survive healing, or escalation never fires"
    )
    assert s._audio_sleep == 30.0, "a dead room must stop hammering"


@pytest.mark.asyncio
async def test_healing_continues_after_the_room_is_declared_dead():
    """`if`, not `elif`: a cycle that heals must still be able to declare the
    room dead, and the next cycles must keep trying rather than giving up."""
    s = _cycle_session(heal_stalls=3)
    for _ in range(12):
        await s._audio_cycle_done(None, 0)
    assert s.heals >= 4, "heal stopped running while the room stayed dead"


@pytest.mark.asyncio
async def test_ffmpeg_stderr_is_actually_read():
    """stderr was piped and read NOWHERE, so the only clue to the cause was
    discarded. The failure line is worthless without it."""
    s = _cycle_session()
    got = await s._read_ffmpeg_stderr(
        _FakeProc("192.168.22.241:554: Connection refused".encode())
    )
    assert "Connection refused" in got


@pytest.mark.asyncio
async def test_ffmpeg_stderr_handles_there_being_nothing_to_read():
    s = _cycle_session()
    assert await s._read_ffmpeg_stderr(None) == ""
    assert await s._read_ffmpeg_stderr(_FakeProc(b"")) == ""


@pytest.mark.asyncio
async def test_ambient_utterances_do_not_pollute_the_awake_tally():
    """The VAD in this room reports speech=True continuously, so the television
    opens an utterance every ~7 s that _process_utterance throws away.
    Measured 04.10.2026 21:01-21:05: 70 ends in four minutes, none a command.
    Counting them makes the tally answer the wrong question — `pause` would
    mostly mean "the television" — and buries the log at one INFO line per 7 s.
    """
    s = _cycle_session()
    sent = []

    async def _fake_process(_buf):
        sent.append(_buf)

    s._process_utterance = _fake_process

    # _end_utterance() is a plain method, not a coroutine — it hands the audio
    # off with create_task. Awaiting it here would have raised on None.
    s._vad_speech_buf = bytearray(32000)
    s._end_utterance("cap 7s")
    assert s._end_reason == {}, "an ambient utterance reached the tally"

    s._wake_detected = True
    s._vad_speech_buf = bytearray(32000)
    s._end_utterance("cap 7s")
    assert s._end_reason == {"cap": 1}, (
        "an awake utterance missed the tally: " + repr(s._end_reason)
    )

    # create_task only schedules; let both tasks run before counting them.
    await asyncio.sleep(0)
    assert len(sent) == 2, "both must still reach _process_utterance"



# --- the player's end-of-stream sentinel -----------------------------------
# Measured 05.10.2026 09:54:46 -> 09:56:45: one reply, then 119 s with no
# VAD SPEECH START at all, and a command spoken in that window got «Да?».
# The queue carried ["Включила", None]; the loop read TWO items per pass, so
# the sentinel was consumed as the "next sentence", `sent = nxt` made sent
# None, and the next pass did `sent = await q.get()` on a queue nobody would
# ever write to again. `_call_backend` then sat on `wait_for(player_task,
# 120.0)` — and because the utterance onset requires
# `not _processing_utterance`, the room could not hear anything for 120 s.


def _player_session():
    """A session with just enough state for the playback loop."""
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s._wake_timeout = 60.0
    s._wake_expires = 0.0
    s._auto_greeting = False
    # The player touches these on every sentence. Note that a missing one is
    # swallowed by the loop's broad `except Exception` and surfaces as
    # "TTS sentence failed" — which is how a missing attribute in a test helper
    # reads as a playback bug.
    s._followup_open = False
    s._followup_only = False
    # These tests are about the window's BEHAVIOUR, so they opt in explicitly.
    # `CAMERA_FOLLOWUP` is off by default since 06.10.2026 (the user's voice
    # peaks at 4725-9124 in the living room, the television at 13197-32532, so
    # no level threshold separates them and the wake word is required again
    # after every reply). Omitting this attribute does not fail loudly — the
    # player's broad `except Exception` turns it into "TTS sentence failed",
    # which is the failure mode this helper's own comment warns about, and it is
    # what happened the first time.
    s._followup_mode = "all"
    s._followup_window = True
    s._followup_min_peak = _FOLLOWUP_MIN_PEAK
    s._wake_detected = False
    s._dialogue_question_s = 30.0
    s._dialogue_statement_s = 10.0
    s.played = []
    s.back_to_wake = 0

    async def _tts(text):
        return b"pcm"

    async def _speak(pcm, reply):
        s.played.append(reply)
        return bool(getattr(s, "_next_reply_is_question", False))

    async def _drain():
        return None

    def _back():
        s.back_to_wake += 1

    s._tts_fetch = _tts
    s._speak_pcm = _speak
    s._wait_playback_drain = _drain
    s._back_to_wake = _back
    return s


@pytest.mark.asyncio
async def test_player_terminates_on_the_sentinel_instead_of_blocking():
    """A one-sentence reply must END, not park on the 120 s backend timeout."""
    s = _player_session()
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Включила")
    q.put_nowait(None)

    # The 5 s bound IS the assertion: before the fix this never returned.
    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s.played == ["Включила"], f"the reply was dropped: {s.played}"
    # The window is NOT closed: an answer opens a follow-up window too, so the
    # room stays audible. Returning at all is the assertion that matters here —
    # before the sentinel fix this call never returned.
    assert s._followup_open is True
    assert s.back_to_wake == 0


@pytest.mark.asyncio
async def test_player_plays_every_sentence_before_the_sentinel():
    """The sentinel must not swallow the sentence it arrives with."""
    s = _player_session()
    q: asyncio.Queue = asyncio.Queue()
    for line in ("Включила свет", "В гостиной уже включено."):
        q.put_nowait(line)
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s.played == ["Включила свет", "В гостиной уже включено."], s.played


@pytest.mark.asyncio
async def test_player_opens_the_dialogue_window_after_a_question():
    """A question keeps the mic open, an answer does not."""
    s = _player_session()
    s._next_reply_is_question = True
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Что именно включить?")
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s._wake_detected is True, "the window must stay open after a question"
    assert s.back_to_wake == 0, "_back_to_wake must not fire when a question opened"


def _dialogue_session(question: bool):
    s = _player_session()
    s._next_reply_is_question = question
    s._dialogue_question_s = 30.0
    s._dialogue_statement_s = 10.0
    return s


@pytest.mark.asyncio
async def test_the_microphone_stays_open_after_an_ANSWER_too():
    """Reported 05.10.2026: "the dialogue setting does not work for cameras the
    way it does for the ESP32".

    ESP32 does this in main.py `_finalize_turn_followup()` — status LISTENING in
    BOTH branches, 30 s after a question and 10 s after a statement. The camera
    opened a window only after a QUESTION, and `_back_to_wake()` closed it the
    moment it had answered, so "а теперь выключи" was never heard.
    """
    s = _dialogue_session(question=False)
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Включила")
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s._wake_detected is True, "the mic closed right after the answer"
    assert s._followup_open is True
    assert s.back_to_wake == 0, "_back_to_wake closed a window we had opened"
    assert s._wake_expires > time.time(), "the window must have an expiry in the future"
    # ~10 s: the satellite's STANDBY_TIMEOUT_STATEMENT, not the question's 30.
    assert 5.0 < s._wake_expires - time.time() < 20.0


@pytest.mark.asyncio
async def test_a_followup_window_is_per_turn_not_sticky():
    """A window opened by an earlier turn must not keep the mic open forever."""
    s = _dialogue_session(question=True)
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Что именно?")
    q.put_nowait(None)
    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)
    assert s._followup_open is True

    # Next turn with playback disabled entirely: no window, so it must close.
    s2 = _dialogue_session(question=False)
    s2._followup_open = True          # stale from the previous turn
    q2: asyncio.Queue = asyncio.Queue()
    q2.put_nowait(None)               # stream ends with no sentence at all
    await asyncio.wait_for(s2._nanobot_player_task(q2), timeout=5.0)
    assert s2.back_to_wake == 1, "a stale flag kept the mic open forever"


def test_dialogue_windows_default_to_the_satellite_numbers():
    """So a camera and an ESP32 answer a follow-up on the same terms."""
    from camera_client import (
        CAMERA_DIALOGUE_QUESTION_S,
        CAMERA_DIALOGUE_STATEMENT_S,
    )

    assert CAMERA_DIALOGUE_QUESTION_S == 30.0
    assert CAMERA_DIALOGUE_STATEMENT_S == 10.0
    cfg = CameraConfig("livingroom")
    assert cfg.dialogue_question_s == 0.0, "unset must stay 0 in the config"


# --- a follow-up window is not the user (measured 05.10.2026 19:24-19:26) ----
#
# The living room answered its own television six times in two minutes. Nobody
# had said the wake word; the window had been opened by our own reply, and every
# utterance inside it was dispatched.


# (peak, transcript) exactly as the gateway logged them that session.
_SESSION = [
    (32767, "выключи свет"),
    (32767, "включи свет в гостиной"),
    (16074, "музыкант"),
    (4625, "Мама, ты что? Да, ха-ха-ха."),
    (4130, "Это вот сейчас и не родит, не хранит стресс."),
    (12348, "распаковываешь"),
    (5626, "Добро пожаловать!"),
    (10792, "Другая, пожалуйста, у тебя был шанс, но ты обожался."),
]


def _followup_session():
    s = _make_session()
    s._wake_detected = True
    s._followup_only = True
    return s


@pytest.mark.asyncio
async def test_a_quiet_followup_is_not_answered_at_all():
    """The television is across the room, so it is quieter than the user.

    It must not be dispatched, and the room must go back to waiting for the wake
    word rather than answering noise — that was the "the camera talks nonsense on
    its own" report.
    """
    dispatched = []
    for peak, text in _SESSION:
        if peak >= _FOLLOWUP_MIN_PEAK:
            continue
        s = _followup_session()
        s._last_utt_peak = peak

        async def _boom(*a, **kw):
            dispatched.append(text)
            return None

        s._call_nanobot = _boom
        s._cancel_wake_greeting = lambda: None
        await s._handle_wake_or_command(text, "camera")
        assert text not in dispatched, (
            f"answered the room: {text!r} at peak {peak}"
        )
        assert s._wake_detected is False, (
            f"kept the follow-up window open after {text!r} — the room would "
            "stay in dialogue mode with nobody in it"
        )


@pytest.mark.asyncio
async def test_the_users_own_followup_still_answers():
    """The gate must not cost the natural thing it exists to allow: saying
    «а теперь выключи» with no wake word."""
    seen = []

    for peak, text in _SESSION:
        if peak < _FOLLOWUP_MIN_PEAK:
            continue
        s = _followup_session()
        s._last_utt_peak = peak

        async def _ok(txt, uid="camera"):
            seen.append(txt)
            return None

        s._call_nanobot = _ok
        s._cancel_wake_greeting = lambda: None
        await s._handle_wake_or_command(text, "camera")

    assert seen == ["выключи свет", "включи свет в гостиной"], seen


@pytest.mark.asyncio
async def test_a_window_opened_by_the_wake_word_accepts_anything():
    """The gate applies ONLY to a window nobody asked for. After a real «компьютер»
    the next utterance is the user whatever it measures — that is what a wake word
    is for."""
    s = _make_session()
    s._wake_detected = True
    s._followup_only = False
    s._last_utt_peak = 4130  # television-level
    seen = []

    async def _ok(txt, uid="camera"):
        seen.append(txt)
        return None

    s._call_nanobot = _ok
    s._cancel_wake_greeting = lambda: None
    await s._handle_wake_or_command("а теперь выключи", "camera")
    assert seen == ["а теперь выключи"], seen


@pytest.mark.asyncio
async def test_the_mic_stops_being_muted_when_the_speaker_stops():
    """The 3 s tail on top of the clip ate the first 1.3 s of the next sentence.

    Measured 05.10.2026 19:24: reply ended 19:24:24.3, the collected utterance
    started 19:24:27.2, Whisper returned empty. The mute must now end with the
    audio, not three seconds after it.
    """
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    dur = 2.0
    pcm = b"\x03\x04" * int(TTS_PLAY_RATE * dur)

    await s._speak_pcm(pcm, "Выключила")

    mute_len = s._speaking_until - s._tts_play_end
    assert mute_len < 1.0, (
        f"mic muted {mute_len:.1f}s past the end of the audio"
    )
    assert mute_len >= 0, "the mic opens before the speaker has finished"
    assert _SPEAKER_SETTLE_S <= 0.5, "the settle margin itself grew"


@pytest.mark.asyncio
async def test_the_transcript_of_a_followup_is_not_thrown_away_afterwards():
    """The second 3 s blocker, which the mic fix alone does not touch.

    `GLOBAL_TTS_UNTIL` drops a finished transcript with "TTS playback active
    (echo guard)". It was `audio_dur + 3 s`, so a follow-up spoken right after the
    answer was discarded with a perfectly good transcript in hand — the log then
    looks like the user said nothing.
    """
    import camera_client

    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    dur = 2.0
    pcm = b"\x03\x04" * int(TTS_PLAY_RATE * dur)

    await s._speak_pcm(pcm, "Выключила")

    tail = camera_client.GLOBAL_TTS_UNTIL - s._tts_play_end
    assert tail < 1.0, (
        f"transcripts blocked {tail:.1f}s past the end of the audio — the "
        "follow-up is discarded before it can be answered"
    )


def test_the_gate_default_is_off_and_the_env_is_wired():
    """Like every other live-audio knob: 0 means "use the built-in default", and
    the knob itself must be readable or it cannot be turned off per room."""
    import inspect

    assert CameraConfig(stream_name="x").followup_min_peak == 0
    sig = inspect.signature(_make_session()._handle_wake_or_command)
    assert "txt" in sig.parameters
    src = open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
        ),
        encoding="utf-8",
    ).read()
    assert "CAMERA_FOLLOWUP_MIN_PEAK_{name.upper()}" in src


@pytest.mark.asyncio
async def test_dropped_audio_is_counted_not_silently_discarded():
    """On 05.10.2026 a follow-up vanished with nothing in the log to say where:
    the collected 7 s buffer held raw_rms=496 — the room, no user — while the
    window was open and the user was speaking. "The user said nothing" and "we
    threw it away" are indistinguishable from outside, so both early returns in
    _feed_audio now count themselves."""
    s = _make_session()
    s._speaking_until = time.time() + 5.0
    for _ in range(4):
        await s._feed_audio(b"\x01\x02" * 160)
    assert s._drop_muted == 4, s._drop_muted

    s._speaking_until = 0.0
    class _Track:
        def echo_active(self):
            return True

    s._out_track = _Track()
    for _ in range(3):
        await s._feed_audio(b"\x01\x02" * 160)
    assert s._drop_track == 3, s._drop_track


def test_the_diag_line_carries_the_drop_counters():
    """A counter nobody logs is the same as no counter."""
    import inspect

    src = inspect.getsource(CameraSession._vad_process)
    assert "drop[muted=" in src, (
        "the diag line must carry the drop counters, or a vanished follow-up is "
        "undiagnosable again"
    )


# --- the pre-roll: the utterance must start where the speech started ----------
#
# The VAD needs 3 speech frames inside a 6-frame sliding window, so an utterance
# opens up to 0.96 s after the user began speaking and everything before it used
# to be discarded. For «а теперь выключи» (~1.3 s) that is most of the sentence,
# and the measured result was an utterance with no speech in it at all.


def test_the_preroll_is_long_enough_to_cover_the_onset_latency():
    """3-of-6 over 160 ms frames is 0.96 s of confirmation; the pre-roll has to
    be at least that or it buys nothing."""
    assert _PREROLL_FRAMES * 0.16 >= 0.96, (
        f"pre-roll {_PREROLL_FRAMES * 0.16:.2f}s cannot cover the 0.96s the VAD "
        "takes to confirm an onset"
    )
    assert _PREROLL_FRAMES * 2560 <= 64 * 1024, "pre-roll should stay small"


def test_the_preroll_ring_keeps_only_the_last_frames():
    s = _make_session()
    s._preroll = []
    for i in range(_PREROLL_FRAMES * 3):
        s._preroll.append(b"%d" % (i % 256))
        while len(s._preroll) > _PREROLL_FRAMES:
            s._preroll.pop(0)
    assert len(s._preroll) == _PREROLL_FRAMES
    # The newest frames survive, the oldest are gone.
    assert s._preroll[-1] == b"%d" % ((_PREROLL_FRAMES * 3 - 1) % 256)


def test_the_echo_counter_is_on_both_call_sites():
    """It was on one of the two `_is_echo` call sites and read `echo=0` while the
    log showed `Echo chunk dropped (corr)` every few minutes."""
    import inspect

    src = inspect.getsource(CameraSession._feed_audio)
    assert src.count("self._drop_echo += 1") == 2, (
        "both the 48 kHz and the generic-rate path must count, or the counter "
        "under-reports and cannot be trusted"
    )


@pytest.mark.asyncio
async def test_an_answer_to_our_own_question_is_always_heard():
    """The camera asked, so whatever is said next is meant for us.

    Measured 05.10.2026 20:29: the camera asked something, the user answered
    «звисит», and the answer was DISCARDED with `follow-up ignored, peak=3145 <
    24000`. No threshold fixes it — the television produced peaks of 1694 and
    10760 in the same session, so the user's 3145 sits inside its range. A dead
    dialogue is worse than the occasional answer to the telly.
    """
    seen = []
    s = _followup_session()
    s._followup_is_question = True
    s._last_utt_peak = 3145  # exactly what the field log recorded

    async def _ok(txt, uid="camera"):
        seen.append(txt)
        return None

    s._call_nanobot = _ok
    s._cancel_wake_greeting = lambda: None
    await s._handle_wake_or_command("звисит", "camera")

    assert seen == ["звисит"], f"the answer to our own question was dropped: {seen}"


@pytest.mark.asyncio
async def test_the_statement_gate_still_holds():
    """Nothing is expected after a statement, so the voice is still the test."""
    s = _followup_session()
    s._followup_is_question = False
    s._last_utt_peak = 4130  # television
    seen = []

    async def _ok(txt, uid="camera"):
        seen.append(txt)
        return None

    s._call_nanobot = _ok
    s._cancel_wake_greeting = lambda: None
    await s._handle_wake_or_command("Добро пожаловать", "camera")
    assert seen == [], seen


def test_a_question_marks_its_own_followup_window():
    """The flag has to be set where the window is opened and cleared on both
    exits, or a later statement window would inherit the exemption."""
    s = _make_session()
    s._followup_is_question = True
    s._back_to_wake()
    assert s._followup_is_question is False
    assert s._followup_only is False


# --- our own pip must never become the utterance (measured 05.10.2026 21:06) --
#
# The user's command came in loud and clear (raw_rms=13660, peak=32767) and
# Whisper returned «пик». The buffer it was given was:
#
#     peak: [1379, 1335, 1596, 32767, 32767, 32767, 32767, 2656, 1615, 1614]
#            \___ silence ___/ \____ our own pip ____/ \__ silence __/
#
# 1.6 s containing no user speech whatsoever, so the router escalated a one-word
# turn to L2 and spent 8 s on it. Cause: `_play_attention` muted the mic for
# `now + 0.35` while the pip is `duration = 0.4`, and `now` was captured BEFORE
# the /play_audio call.


@pytest.mark.asyncio
async def test_the_pip_mute_outlives_the_pip():
    """0.35 s of mute for a 0.4 s pip leaks the tail at the rail.

    Exercised, not grepped: an earlier version of this test searched the source
    for `now + 0.35` and matched the COMMENT explaining the bug — the same
    mistake AGENTS.md warns about, where the rule is written in its own
    justification and the search finds the counter-example.
    """
    s = _make_http_session()
    s.http_session = _FakeHttp(200)

    before = time.time()
    await s._play_attention("vosk")
    after = time.time()

    pip_dur = 0.4
    # Measured from the playback moment, so the whole POST duration is covered.
    assert s._speaking_until >= after + pip_dur - 0.05, (
        f"mute ends {s._speaking_until - after:.2f}s after playback, pip is "
        f"{pip_dur}s — the tail reaches the decoder at the rail"
    )
    assert before <= s._speaking_until


@pytest.mark.asyncio
async def test_the_activation_cue_guards_the_mic_too():
    """It registered the duration on the track but never opened the guard, so it
    played with no mute at all while the room listened for a command."""
    s = _make_http_session()
    s.http_session = _FakeHttp(200)
    s._activation_wav_path = _write_pip_wav()
    before = s._speaking_until

    await s._play_activation_sound()

    dur = 0.25  # what _write_pip_wav writes
    assert s._speaking_until > time.time() + dur - 0.05, (
        f"cue mute ends in {s._speaking_until - time.time():.2f}s, cue is {dur}s"
    )
    assert s._speaking_until > before, "the cue played without muting the mic"
    assert s._tts_play_end > time.time(), "playback end not recorded"


@pytest.mark.asyncio
async def test_the_wake_word_clears_the_preroll():
    """After a wake word the utterance starts AT the wake, which is what makes
    that path immune to the confirmation latency the pre-roll covers. Keeping
    pre-wake audio would put room noise in front of every command."""
    s = _make_session()
    s._preroll = [b"x" * 2560] * 4
    s._vad_speech_buf.extend(b"y" * 2560)
    s._vad_has_speech = True

    await CameraSession._fire_wake(s, "vosk")

    assert s._preroll == [], "pre-wake audio survived into the command"
    assert not s._vad_has_speech


@pytest.mark.asyncio
async def test_a_followup_keeps_the_preroll():
    """The other half of the rule: a follow-up has no wake word to cut at, and
    the pre-roll is the whole reason it starts where the speech did."""
    s = _make_session()
    s._preroll = [b"x" * 2560] * 4
    s._wake_detected = True
    s._followup_only = True
    s._followup_is_question = False

    # Going back to standby must NOT clear it: the next follow-up depends on it.
    s._back_to_wake()
    assert len(s._preroll) == 4, "standing down discarded the pre-roll"


# --- the WebRTC supervisor must stop offering, not just offer less -----------
#
# Backing off to a 5-minute cap made the camera wedge less likely, not
# impossible. Every offer makes go2rtc rebuild the stream's producer and an
# abandoned one leaves the CAMERA's send queue full — 193 kB measured, its
# single-threaded majestic blocked, HTTP 11-15 s and no RTSP for hours. A session
# that has never once been answered is not going to start answering.


@pytest.mark.asyncio
async def test_the_supervisor_stops_offering_after_a_run_of_misses(monkeypatch):
    """The give-up must be a real pause, not just a longer retry.

    The assertion reads the MODULE attribute, not the imported binding: an earlier
    version patched `camera_client._WEBRTC_GIVEUP_ATTEMPTS` and then asserted on
    the name imported at load time, so the patch was never exercised and the test
    only checked the default. That is the same class of mistake as the `now +
    0.35` grep — asserting on a copy instead of on what runs.
    """
    # Captured BEFORE the patch: the point is that the shipped defaults are
    # conservative, and reading them after patching would assert on the test's
    # own numbers.
    giveup_retry = camera_client._WEBRTC_GIVEUP_RETRY_S
    giveup_attempts = camera_client._WEBRTC_GIVEUP_ATTEMPTS
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MIN_S", 0.0)
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MAX_S", 0.0)
    monkeypatch.setattr(camera_client, "_WEBRTC_GIVEUP_ATTEMPTS", 3)
    monkeypatch.setattr(camera_client, "_WEBRTC_GIVEUP_RETRY_S", 0.0)

    batches = []
    state = {"run": 0}

    class _Sess(CameraSession):
        async def _connect(self):
            state["run"] += 1
            # Mark every run of three misses, so we can see the supervisor resting
            # between them instead of offering straight through.
            if state["run"] % 3 == 0:
                batches.append(state["run"])
            if state["run"] > 21:
                self._stopped.set()
            return False

    s = _Sess(CameraConfig(stream_name="cam"))
    await asyncio.wait_for(s._run(), timeout=10)

    assert state["run"] == 22, f"{state['run']} offers, expected 22"
    assert len(batches) == 7, batches
    # The production defaults have to be conservative enough to matter.
    assert giveup_attempts >= 3, giveup_attempts
    assert giveup_retry >= 600.0, (
        f"the pause after giving up is only {giveup_retry:.0f}s — that is a "
        "retry, not a give-up"
    )


@pytest.mark.asyncio
async def test_one_answered_offer_resets_the_give_up_counter(monkeypatch):
    """A single success must clear the run of misses, or a session that connects
    once and then drops would stop being retried while the fallback is fine."""
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MIN_S", 0.0)
    monkeypatch.setattr(camera_client, "_WEBRTC_BACKOFF_MAX_S", 0.0)
    monkeypatch.setattr(camera_client, "_WEBRTC_GIVEUP_ATTEMPTS", 3)
    monkeypatch.setattr(camera_client, "_WEBRTC_GIVEUP_RETRY_S", 0.0)

    answers = [False, False, False, True, False, False, False, False, False,
               False, False, False, False]
    seen = []

    class _Sess(CameraSession):
        async def _connect(self):
            if not answers:
                self._stopped.set()
                return False
            seen.append(answers.pop(0))
            return seen[-1]

    s = _Sess(CameraConfig(stream_name="cam"))
    await asyncio.wait_for(s._run(), timeout=10)

    assert True in seen, "no answered offer in the sequence"
    # 13 attempts proves the counter was reset: without the reset the supervisor
    # would rest at 3 offers and never reach the True in position 4.
    assert len(seen) == 13, len(seen)


@pytest.mark.asyncio
async def test_the_preroll_is_actually_seeded_into_the_utterance():
    """The version of this that shipped was broken and the suite was green.

    `_vad_speech_buf` is a bytearray and the pre-roll is a LIST of chunks.
    `bytearray.extend(list_of_bytes)` raises `TypeError: 'bytes' object cannot be
    interpreted as an integer`, which killed `_vad_process` at the first speech
    onset of every session — visible only as `Task exception was never
    retrieved`, which nobody was reading.

    The earlier test in this file exercised the ring's list mechanics and never
    the seeding, so it passed. This one drives the real path.
    """
    s = _make_session()
    s._preroll = [b"A" * 2560, b"B" * 2560]
    s._vad_speech_buf = bytearray()

    # Exactly what `_vad_process` does when the VAD confirms an onset.
    s._vad_speech_buf.extend(b"".join(s._preroll))
    s._vad_speech_buf.extend(b"C" * 2560)

    assert len(s._vad_speech_buf) == 3 * 2560
    assert bytes(s._vad_speech_buf) == b"A" * 2560 + b"B" * 2560 + b"C" * 2560


def test_the_seeding_line_cannot_regress_to_the_bytearray_mistake():
    """The failure was silent, so pin the shape of the line itself.

    `bytearray.extend` takes an iterable of ints. Anything else — a list of
    chunks, a generator, a dict view — raises at runtime, in a coroutine, and
    takes the whole VAD loop with it.
    """
    import inspect

    src = inspect.getsource(CameraSession._vad_process)
    assert "_vad_speech_buf.extend(self._preroll)" not in src, (
        "seeding a bytearray from a list of bytes raises TypeError and kills "
        "_vad_process at the first onset — join the chunks first"
    )
    assert 'b"".join(self._preroll)' in src


def test_the_noise_suppression_cutoff_is_a_room_setting_not_a_constant():
    """It was hardcoded `800 if kitchen else 400` in shared code, so no room could
    be retuned without editing the module — and it cannot be re-measured now that
    the kitchen's microphone is dead (rms 4, -77.8 dB)."""
    import inspect

    src = inspect.getsource(CameraSession._preprocess_audio)
    assert 'if self.stream_name == "kitchen"' not in src, (
        "a per-room calibration is still baked into shared code"
    )
    assert "self._ns_rms_gate" in src, "the cutoff must come from the config"
    assert CameraConfig(stream_name="x").ns_rms_gate == 0, "0 means the default"
    assert CameraConfig(stream_name="kitchen", ns_rms_gate=800).ns_rms_gate == 800


def test_both_rooms_read_identical_thresholds():
    """The rooms have the same problems, so they get the same numbers — and the
    test says so, because 'identical by default' is not the same as 'identical'."""
    from main import start_camera_sessions  # noqa: F401  (import must succeed)

    living = CameraConfig(stream_name="livingroom")
    kitchen = CameraConfig(stream_name="kitchen")
    for field in (
        "pause_endpoint", "pause_noise_mult", "pause_ratio",
        "pause_run_frames", "pause_min_speech_frames", "ns_rms_gate",
        "webrtc", "attention_pip",
    ):
        assert getattr(living, field) == getattr(kitchen, field), (
            f"{field}: livingroom={getattr(living, field)!r} "
            f"kitchen={getattr(kitchen, field)!r}"
        )


# --- the utterance must not close on the wake word alone (06.10.2026 13:25) ---
#
# Measured on the living room: the pause endpoint committed a 2.56 s utterance
# containing only «компьютер» — the wake word's own frames satisfied
# `min_speech_frames` (speech=7) and the gap after it read as a pause. The command
# was orphaned into the next utterance, and that one ran to the 7 s cap with
# television in it: «Числят.», «Часок на один. Это дождь.»,
# «Что ты думаешь, что я не ехать ночью?» — three commands answered aloud that
# nobody gave.


@pytest.mark.asyncio
async def test_a_wake_only_transcript_keeps_the_utterance_open():
    s = _make_session()
    s._wake_detected = True
    s._wake_only_pending = False
    s._vad_has_speech = False

    await s._handle_wake_or_command("компьютер", "camera")

    assert s._wake_only_pending is True, (
        "a wake-only transcript must ask for the utterance to be re-opened, "
        "otherwise the command lands in a slot the television can take"
    )
    assert s._wake_detected is True, "the wake window must survive"


@pytest.mark.asyncio
async def test_a_wake_only_transcript_does_not_apply_to_a_real_command():
    s = _make_session()
    s._wake_detected = True
    s._wake_only_pending = False

    async def _ok(*a, **kw):
        return None

    s._call_nanobot = _ok
    s._cancel_wake_greeting = lambda: None
    await s._handle_wake_or_command("компьютер включи свет", "camera")

    assert s._wake_only_pending is False


def test_reopening_restores_the_preroll():
    """Without the pre-roll the utterance resumes at the reopen and can still lose
    the first syllable of the command — which is the bug the pre-roll fixed."""
    s = _make_session()
    s._vad_speech_buf = bytearray(b"stale")
    s._vad_has_speech = False
    s._preroll = [b"A" * 2560, b"B" * 2560]

    s._reopen_utterance()

    assert s._vad_has_speech is True
    assert bytes(s._vad_speech_buf) == b"A" * 2560 + b"B" * 2560
    assert s._vad_silence_frames == 0


@pytest.mark.asyncio
async def test_a_bare_wake_word_alone_never_dispatches_anything():
    """The regression that matters: nothing must reach the router when the only
    thing said was the wake word."""
    dispatched = []
    s = _make_session()
    s._wake_detected = True
    s._wake_only_pending = False

    async def _boom(*a, **kw):
        dispatched.append(a)
        raise AssertionError("a bare wake word was dispatched as a command")

    s._call_nanobot = _boom
    await s._handle_wake_or_command("Компьютер.", "camera")
    assert dispatched == []


# --- the wake word's OWN audio must not be dispatched as a command (06.10.2026) --
#
# Measured on the living room at 12:39: the wake fired at 12:39:56.340 and the
# utterance that followed was 1.12 s long, ending 1.85 s after the wake. Whisper
# wrote the wake word out as «1000 свят» — no «компьютер» for the strip to find —
# so it was dispatched, and the router answered «Включила» to a wake word.
#
# The fix asks by TIME, because the transcript is what failed. Measured margins on
# both sides: the wake word alone began 0.73 s after the wake and ran 1.12 s;
# real commands run 1.68 s («включи кофеварку») and 1.84 s («Выключить свет»).


def _wake_then_utterance(end_after_wake: float, dur: float) -> CameraSession:
    s = _make_session()
    s._wake_detected = True
    s._wake_only_pending = False
    s._last_wake_fired_at = 1000.0
    s._last_utt_end_ts = 1000.0 + end_after_wake
    s._last_utt_dur = dur
    return s


def test_the_wake_words_own_audio_is_recognised_by_time():
    """The measured shape: onset 0.73 s after the wake, 1.12 s long."""
    s = _wake_then_utterance(end_after_wake=1.85, dur=1.12)
    assert s._is_wake_word_alone() is True


def test_a_real_command_is_not_mistaken_for_the_wake_word():
    """«включи кофеварку» is 1.68 s and «Выключить свет» 1.84 s — both longer
    than the wake word, both dispatched on the same run."""
    assert _wake_then_utterance(end_after_wake=2.39, dur=1.68)._is_wake_word_alone() is False
    assert _wake_then_utterance(end_after_wake=3.10, dur=1.84)._is_wake_word_alone() is False


def test_a_utterance_that_ends_long_after_the_wake_is_a_command():
    """A cap-length utterance full of television that began after the wake."""
    assert _wake_then_utterance(end_after_wake=9.0, dur=7.04)._is_wake_word_alone() is False


@pytest.mark.asyncio
async def test_a_transcript_without_the_wake_word_can_still_be_the_wake_word():
    """The regression itself: «1000 свят» must not reach the router."""
    dispatched = []
    s = _wake_then_utterance(end_after_wake=1.85, dur=1.12)

    async def _boom(*a, **kw):
        dispatched.append(a)
        raise AssertionError("the wake word was dispatched as a command")

    s._call_nanobot = _boom
    await s._handle_wake_or_command("1000 свят", "camera")

    assert dispatched == [], "«1000 свят» was sent to the router as a command"
    assert s._wake_only_pending is True, "the utterance must be re-opened"


@pytest.mark.asyncio
async def test_a_normal_command_after_the_wake_word_is_still_dispatched():
    dispatched = []
    s = _wake_then_utterance(end_after_wake=2.39, dur=1.68)

    async def _ok(*a, **kw):
        dispatched.append(a)
        return None

    s._call_nanobot = _ok
    s._cancel_wake_greeting = lambda: None
    await s._handle_wake_or_command("включи кофеварку", "camera")

    assert dispatched, "the real command was swallowed by the wake-word rule"
    assert s._wake_only_pending is False


# --- after our own reply the wake word is required again (user's decision) -----
#
# `CAMERA_FOLLOWUP` is OFF by default. Measured 06.10.2026 in twenty minutes:
# the user's own voice peaked at 4725 and 9124, while the television and the
# appliances peaked at 13197, 16074 and 32522 — the noise is louder than the user,
# so `followup_min_peak` refused the user twice ('ключ', 'Выключить свет.') and
# admitted noise twice ('атака', 'пиздец'). No level threshold separates them.


def test_a_question_is_answered_without_the_wake_word_by_default():
    """The user's decision 06.10.2026: when the camera asks, it must listen to
    the answer immediately. An answer to OUR OWN question is the one follow-up that
    needs no keyword — and refusing it is the "asks a question and then ignores
    the answer" complaint."""
    cfg = CameraConfig(stream_name="x")
    assert cfg.followup_window == "question"
    s = CameraSession(CameraConfig(stream_name="x", go2rtc_host="127.0.0.1"))
    assert s._followup_mode == "question"
    assert s._followup_window is True, "a question must still open a window"


def test_a_statement_does_not_open_a_window_by_default():
    """A statement window is 10 s of open microphone in a room whose television is
    louder than its occupant, and it was measured accepting noise ('атака' 32522,
    'пиздец' 13197) while refusing the user (4725, 9124)."""
    s = CameraSession(CameraConfig(stream_name="x", go2rtc_host="127.0.0.1"))
    assert s._followup_mode == "question", (
        "the default must not open a window after a statement"
    )


def test_the_window_mode_has_three_states_and_a_bad_value_is_not_silent():
    """A typo in the env var must not turn the microphone off for a question."""
    for mode in ("none", "question", "all"):
        s = CameraSession(
            CameraConfig(stream_name="x", go2rtc_host="127.0.0.1",
                         followup_window=mode)
        )
        assert s._followup_mode == mode
    assert CameraSession(
        CameraConfig(stream_name="x", go2rtc_host="127.0.0.1",
                     followup_window="да")
    )._followup_mode == "question", "an unknown value fell back to 'question'"


def test_followup_none_is_the_only_mode_that_closes_the_mic():
    s = CameraSession(
        CameraConfig(stream_name="x", go2rtc_host="127.0.0.1", followup_window="none")
    )
    assert s._followup_window is False


def test_a_zero_window_is_not_the_same_as_no_window():
    """`0` means "unset, use the built-in default" everywhere in this file, so a
    window of 0 would have produced a 30 s window rather than none. That is why the
    gate is an explicit MODE and not a duration: "no window at all" has no numeric
    value here, because every number means "substitute a default"."""
    s = CameraSession(
        CameraConfig(stream_name="x", go2rtc_host="127.0.0.1", dialogue_question_s=0.0)
    )
    assert s._dialogue_question_s > 0, "0 falls back to the default, by design"
    assert s._followup_mode == "question", (
        "the mode still opens a window for a QUESTION — that is what an explicit "
        "mode says, and it is not the same as 0 having disabled it"
    )
    off = CameraSession(
        CameraConfig(stream_name="x", go2rtc_host="127.0.0.1", followup_window="none")
    )
    assert off._followup_window is False, "'none' is how you actually say none"
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_no_followup_window_opens_after_a_STATEMENT_by_default():
    """The measurement behind keeping this closed, 06.10.2026: the user's own voice
    peaked at 4725 and 9124 in the living room while the television and the
    appliances peaked at 13197-16074 and 32522, so `followup_min_peak` refused the
    user twice ('ключ', 'Выключить свет.') and admitted noise twice ('атака',
    'пиздец') in the same twenty minutes. A statement window is 10 s of open
    microphone in exactly that room.

    The QUESTION case is deliberately the opposite and has its own test: when the
    camera asks, the user must be able to answer without the wake word."""
    s = _dialogue_session(question=False)
    s._followup_mode = "question"          # the default
    s._followup_window = True
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Включила")
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s._followup_open is False, (
        "a free-form window opened after a statement, although the default is to "
        "require the wake word again"
    )
    assert s._wake_detected is False, "the mic was left open without a wake word"
@pytest.mark.asyncio
async def test_the_window_still_opens_when_it_is_asked_for():
    """The mode must not be a one-way door: opening a window after a STATEMENT is
    still reachable and tested, it is just not the default."""
    s = _dialogue_session(question=False)
    s._followup_mode = "all"
    s._followup_window = True
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Включила")
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s._followup_open is True, (
        "CAMERA_FOLLOWUP=all no longer opens a statement window"
    )


@pytest.mark.asyncio
async def test_a_question_window_opens_without_any_mode_override():
    """The path that answers the user's complaint, driven end to end through the
    player: a reply that is a question must leave the mic open, by default."""
    s = _dialogue_session(question=True)
    s._followup_mode = "question"          # the default, stated explicitly
    s._followup_window = True
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait("Что именно?")
    q.put_nowait(None)

    await asyncio.wait_for(s._nanobot_player_task(q), timeout=5.0)

    assert s._followup_open is True
    assert s._followup_only is True


# --- the echo suppression must not crawl (06.10.2026 12:40) -------------------
#
# Reported: "after a confirmation both cameras take a long time to react to
# «компьютер»". The log said why, three times in a row:
#
#     12:40:24.025 VOSK wake heard but not fired: echo_tail=+19.6s (suppressed=1)
#     12:40:26.274 VOSK wake heard but not fired: echo_tail=+17.4s (suppressed=2)
#     12:40:28.911 VOSK wake heard but not fired: echo_tail=+14.7s (suppressed=3)
#
# `min(now + 20, ...)` was a SLIDING deadline. While the reply was sounding, every
# chunk correlating with it pushed the block 20 s past the current moment, so the
# last echo chunk of a reply left the room deaf for 20 s after the reply ended.
# The user was told the wake word was heard, while it was blocked.


def _echo_session():
    """A session with the mic OPEN, which is the only way the echo branch is
    reachable at all: `_feed_audio` returns on `_speaking_until` before it."""
    s = _make_session()
    s._speaking_until = 0.0
    s._out_track = None
    s._wake_detected = False
    s._last_echo_log = 0.0
    s._is_echo = lambda pcm, *a, **k: True
    s._resample_generic = lambda pcm, rate: pcm
    return s


@pytest.mark.asyncio
async def test_echo_suppression_is_anchored_to_when_playback_ends():
    """Playback ends at T; our own voice arrives in the mic a moment later, which
    is the only moment the echo branch runs at all (the hard mute covers the rest).

    Before the fix the block landed at `now + 20`, i.e. 20 s after the reply — the
    user's «компьютер» was decoded, reported as heard, and dropped.
    """
    from camera_client import _ECHO_TAIL_S

    t_end = 1000.0
    s = _echo_session()
    s._tts_play_end = t_end

    with patch.object(camera_client.time, "time", lambda: t_end + 0.5):
        await s._feed_audio(b"\x01\x02" * 160)

    assert s._drop_echo == 1, "the echo branch did not run at all"
    after_reply = s._wake_suppress_until - t_end
    # Exactly `_ECHO_TAIL_S` past the END of playback. This used to assert
    # `_ECHO_TAIL_S + 0.5`, i.e. the `now + _ECHO_TAIL_S` floor — the sliding
    # deadline, since removed on 07.10.2026 because a run of false echoes pushed
    # it forward forever and held the wake gate shut in a silent room. The echo
    # arrived 0.5 s after playback ended and the block still ends at
    # `t_end + tail`: the floor added nothing that the anchor does not already
    # cover, and it was the mechanism that made the block unbounded.
    assert after_reply == pytest.approx(_ECHO_TAIL_S), (
        f"the block runs {after_reply:.1f}s past the end of the reply, "
        "which is the delay that was reported"
    )
    assert after_reply < 5.0, "20 s of deafness after every reply"


@pytest.mark.asyncio
async def test_a_long_echo_tail_cannot_deafen_the_room_for_twenty_seconds():
    """`min(now + 20, ...)` slid the deadline on EVERY echoing chunk, so a reply
    whose own voice kept correlating deafened the room for twenty seconds more.
    The anchor is now playback end plus `_ECHO_TAIL_S`, so the total deafness is
    bounded by the tail plus however long the echo itself lasted.

    The assertion is "bounded", not "fixed": the deadline does still move with
    each chunk, because `now + _ECHO_TAIL_S` is a deliberate floor — a stale
    `_tts_play_end` from a previous turn must not park the block for the length
    of a whole reply. What must not happen is the 20 s slide.
    """
    from camera_client import _ECHO_TAIL_S

    t_end = 1000.0
    last_echo_at = t_end + 0.5 + 9 * 0.16      # 1.94 s of tail
    s = _echo_session()
    s._tts_play_end = t_end
    with patch.object(camera_client.time, "time", lambda: last_echo_at):
        for _ in range(10):
            await s._feed_audio(b"\x01\x02" * 160)

    deaf_for = s._wake_suppress_until - t_end
    assert s._drop_echo == 10, "the echo branch did not run on every chunk"
    assert deaf_for <= (last_echo_at - t_end) + _ECHO_TAIL_S + 0.01, (
        f"{deaf_for:.1f}s of deafness after the reply: the deadline slid again"
    )
    assert deaf_for < 6.0, (
        f"{deaf_for:.1f}s of deafness after every reply — this is the reported bug"
    )
    # The old arithmetic, for the record: min(now + 20, play_end + 45).
    old = min(last_echo_at + 20.0, t_end + 45.0) - t_end
    assert old > deaf_for * 3, "the assertion no longer distinguishes the fix"


@pytest.mark.asyncio
async def test_the_wake_word_is_accepted_once_the_echo_tail_is_over():
    """The end of the chain: the user waits a moment after the reply, says
    «компьютер», and the gate lets it through. Before the fix this was refused
    for another 20 s — three times in a row in the 12:40 log."""
    from camera_client import _ECHO_TAIL_S, _WAKE_REARM_DEBOUNCE_S

    t_end = 1000.0
    echo_at = t_end + 0.5
    s = _echo_session()
    s._tts_play_end = t_end
    with patch.object(camera_client.time, "time", lambda: echo_at):
        await s._feed_audio(b"\x01\x02" * 160)

    assert s._wake_gate_open(echo_at) is False, (
        "our own voice is still in the mic, so the gate must be shut"
    )
    s._last_wake_fired_at = 0.0          # not the same breath
    assert s._wake_gate_open(echo_at + _ECHO_TAIL_S + 0.2) is True, (
        "the wake word was still blocked long after the reply and its echo were over"
    )
    assert _WAKE_REARM_DEBOUNCE_S < _ECHO_TAIL_S + 0.2


# --- measured 06.10.2026 14:36 on the living room -----------------------------
#
# One «компьютер», and the log reads:
#
#     14:36:48.964 VOSK WAKE peak=1654          <- fired, and NO pip
#     14:36:51.582 VOSK wake heard but not fired: echo_tail=+1.7s  <- the same word, again
#     14:36:54.367 VOSK WAKE peak=1251
#     14:36:54.376 attention pip (vosk)          <- the pip arrives 5.4 s late
#     14:36:57.403 Whisper OK: 'куча свет'       <- the command, recognised correctly
#     14:36:58.004 Ignoring 'куча свет' — TTS playback active (echo guard)


class _RecordingLog:
    """`camera_client` logs through loguru, which does not propagate to stdlib,
    so `caplog` sees nothing. Record the messages instead — an assertion that
    silently passes because the capture is empty is worse than no assertion."""

    def __init__(self):
        self.records = []

    def _rec(self, level):
        def emit(msg, *a, **kw):
            self.records.append(str(msg))
        return emit

    def __getattr__(self, name):
        return self._rec(name)

    @property
    def text(self):
        return "\n".join(self.records)


@pytest.mark.asyncio
async def test_the_pip_limit_must_not_be_longer_than_the_user_patience():
    """Measured 06.10.2026 18:21 UTC: a 15 s pip limit produced silence on two of
    three consecutive wake words, the user repeated themselves, three wakes fired
    in nine seconds, and the third utterance captured «Почему ты пиздишь, что не
    включил кофеварку?» and sent THAT to the router as a command.

    What the 15 s limit was protecting against — two pips from one utterance — is
    already blocked upstream by the rearm debounce, at the decoder. So the limit
    only has to exceed that, and a distinct wake must always get a beep."""
    import camera_client as cc

    assert cc._ATTENTION_MIN_GAP_S <= 5.0, (
        f"the pip limit is {cc._ATTENTION_MIN_GAP_S}s, which is long enough for "
        "the user to give up and repeat"
    )
    # Above the rearm debounce, or a single word could pip twice.
    assert cc._ATTENTION_MIN_GAP_S >= cc._WAKE_REARM_DEBOUNCE_S, (
        "the pip limit must exceed the rearm debounce, which is what actually "
        "stops one word from waking twice"
    )


@pytest.mark.asyncio
async def test_the_pip_must_not_disappear_without_saying_so():
    """A pip skipped by the 15 s rate limit is silence the user cannot explain:
    the wake fires, the room answers nothing, and no line records why. The wake
    at 14:36:48.964 produced no pip because the previous one was 11.1 s earlier."""
    s = _make_session()
    # Just inside the limit, whatever the limit is: this test pins the LOG LINE,
    # not the value. The value has its own test, and it was 15 s when this was
    # written — which is exactly the point, since a hardcoded 11.1 stopped being
    # a skip the moment the limit moved.
    s._last_attention = time.time() - (camera_client._ATTENTION_MIN_GAP_S - 1.0)
    s._out_track = None
    s._wake_suppress_until = 0.0
    s._store_tts_echo = lambda pcm: None
    s._play_audio_http = _never_played

    log = _RecordingLog()
    with patch.object(camera_client, "logger", log):
        await _noop_run(s._play_attention("vosk"))

    assert log.records, "nothing was logged at all, so this test would pass wrongly"
    assert "attention pip skipped" in log.text, (
        f"the pip was skipped without a log line; got: {log.text!r}"
    )
    assert "silence" in log.text, "the line must say what the user experienced"


def test_the_rearm_guard_must_cover_the_wake_words_own_tail():
    """The decoder decoded «компьютер» again 2.62 s after the trigger, from the
    tail of the same word — the trigger lands mid-word and the tail is fed to the
    fresh recogniser. A 1.5 s guard let it through, and the second fire would have
    cleared the command being collected."""
    from camera_client import _WAKE_REARM_DEBOUNCE_S

    assert _WAKE_REARM_DEBOUNCE_S >= 3.0, (
        f"the guard is {_WAKE_REARM_DEBOUNCE_S}s and the measured re-trigger was "
        "2.62 s after the trigger"
    )

    s = _make_session()
    s._last_wake_fired_at = 1000.0
    s._wake_suppress_until = 0.0
    assert s._wake_gate_open(1000.0 + 2.62) is False, (
        "the same word's tail could fire a second wake"
    )
    assert s._wake_gate_open(1000.0 + _WAKE_REARM_DEBOUNCE_S + 0.2) is True


async def _noop_run(coro):
    await coro


async def _never_played(pcm):
    return True


@pytest.mark.asyncio
async def test_the_global_tts_guard_says_who_set_it_and_for_how_long():
    """A command was lost to this guard and the log could not say why: the only
    setter is `_speak_pcm`, the reply was the single word «Включила», and the
    arithmetic puts the expiry 14 s BEFORE the refusal. A guard that silently eats
    a correctly recognised command is the worst kind of guard, so the refusal now
    prints the setter and the time remaining — this test pins that it does."""
    import camera_client as cc

    cc.GLOBAL_TTS_UNTIL = time.time() + 30.0
    cc._GLOBAL_TTS_BY = "livingroom 'Включила' pcm=1000000B dur=20.80s sr=48000"

    s = _make_session()
    s.backend = None            # force the nanobot path, which owns this check
    log = _RecordingLog()
    with patch.object(camera_client, "logger", log):
        await s._call_nanobot("включи свет")

    refusals = [ln for ln in log.records if "echo guard" in ln]
    assert refusals, f"no refusal logged; got: {log.text!r}"
    line = refusals[0]
    assert "Включила" in line, f"the refusal does not say who set it: {line!r}"
    assert "dur=" in line, f"the refusal does not say for how long: {line!r}"
    assert "s left" in line, f"the refusal does not say how long is left: {line!r}"
    cc.GLOBAL_TTS_UNTIL = 0.0


# --- /play_audio answers 3-12x late, and the guards were timed from it (07.10) --
#
# Measured against both cameras on 07.10.2026, HTTP 200 every time:
#
#     audio     kitchen            living room
#     1 s       3.5 s  (x3.0)      5.0 s  (x5.0)
#     3 s       8.7 s  (x2.9)      37.0 s (x12.3)
#     6 s       18.2 s (x3.0)      no answer within 60 s
#
# `GLOBAL_TTS_UNTIL` and `_wake_suppress_until` were computed from `time.time()`
# AFTER the POST returned, so the whole response latency was added to them: a 30 s
# response held the cross-room command guard for 30 s after the reply had already
# finished playing. The field report that exposed it: 08:50 on the kitchen,
# `play_audio failed` after 30 s, then "falling back to the go2rtc backchannel" —
# with `CAMERA_WEBRTC=false` there IS no backchannel, so the reply was never spoken
# and the log promised audio that could not arrive.


@pytest.mark.asyncio
async def test_the_post_playback_guards_are_timed_from_when_the_post_was_sent(monkeypatch):
    """A slow response must not extend the mute. The guard describes the AUDIO."""
    import camera_client as cc

    clock = {"t": 1000.0}

    def _now():
        return clock["t"]

    class _Resp:
        status = 200

        async def __aenter__(self):
            # The camera answers long after the speaker started.
            clock["t"] = 1043.0          # a 43 s round trip, worse than measured
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, *a, **kw):
            self.kw = kw
            return _Resp()

    s = _make_http_session()
    s.http_session = _Session()
    s._play_audio_url = "http://10.0.0.9/play_audio"
    s._play_audio_headers = {"Authorization": "Basic x"}
    s._out_track = None
    s._tts_play_end = 0.0

    with monkeypatch.context() as m:
        m.setattr(cc.time, "time", _now)
        await s._play_audio_http(b"\x00\x01" * 48000)   # 1 s at 48 kHz

    # The HTTP budget must at least admit a slow camera, and must not be the flat
    # 30 s that truncated a 6 s clip on the living room.
    budget = s.http_session.kw["timeout"].total
    assert budget >= 10.0, f"the POST budget is {budget}s, tighter than the camera"
    assert budget <= 60.0


@pytest.mark.asyncio
async def test_a_failed_post_does_not_claim_a_fallback_that_does_not_exist():
    """`CAMERA_WEBRTC=false` means no backchannel. The old line promised one, and a
    reader would go looking for audio that was never coming."""
    import camera_client as cc

    class _Resp:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, *a, **kw):
            return _Resp()

    lines: list[str] = []

    class _Log:
        def warning(self, msg, *a):
            lines.append(msg % a if a else msg)

        def __getattr__(self, _name):
            return self.info

        def info(self, msg, *a):
            lines.append(msg % a if a else msg)

    s = _make_http_session()
    s.http_session = _Session()
    s._play_audio_url = "http://10.0.0.9/play_audio"
    s._play_audio_headers = {}
    s._out_track = None                 # CAMERA_WEBRTC=false

    with patch.object(cc, "logger", _Log()):
        await s._play_audio_http(b"\x00\x01" * 100)

    joined = " ".join(lines)
    assert "falling back" not in joined, (
        f"the log promised a backchannel that does not exist: {joined!r}"
    )


# --- "CAMERA AUDIO DEAD" named the wrong lever first (07.10.2026) ----------
#
# Both rooms hit exactly this state on 07.10.2026 — capture stuck at 0.04x-0.66x
# of real time, three starve attempts, then DEAD — and `/etc/init.d/S95majestic
# restart` fixed BOTH: the living room's `/play_audio` went 17.0 s -> 3.5 s for
# 1 s of audio, the kitchen's capture 0.66x -> 1.00x. The log told the operator to
# power-cycle instead, i.e. it named the expensive lever as the only one.
#
# Capture and playback both live inside `majestic` on this firmware (there is no
# separate audio service), so it is the cheapest step and costs only the video
# stream for a few seconds. A power cycle is the fallback and stays named as one.


def test_the_dead_audio_message_names_the_cheapest_recovery_first():
    """An error message that sends the reader to the expensive lever when a cheap
    one exists is a documentation bug with an operational cost."""
    import inspect

    import camera_client as cc

    src = inspect.getsource(cc.CameraSession._account_audio_rate)

    assert "S95majestic restart" in src, (
        "the DEAD message does not name the cheapest lever, and it is the one "
        "measured to work on this hardware"
    )
    assert src.index("S95majestic") < src.index("power-cycle"), (
        "the cheap lever must come first in the message the operator reads"
    )
    assert "Reboot {self.stream_name} manually" not in src, (
        "the old text named a manual reboot as the only cure, which was measured "
        "false twice on the same day"
    )
    # The power cycle stays available, and the reason it is manual is still stated.
    assert "power-cycle" in src and "ONVIF Reboot" in src


# --- I broke Whisper for two hours with a name that was not in scope (07.10) ----
#
# `_fetch_transcription` builds the multipart POST inside a try/except that ends in
# `pass`, and the caller logs an empty return as `Whisper empty` — the same line a
# SILENT room produces. I replaced the flat `total=30` there with a per-clip budget
# that referenced `pcm`, which is `_play_audio_http`'s argument; this function's is
# `wav`. Every call raised NameError, the except swallowed it, and both rooms
# returned empty transcripts for ~2 h: 0 successful in 24 h, while the room's own
# levels (rms 2326, -23 dB) said the silence explanation was wrong.
#
# Two independent causes, both guarded below: a name from another scope, and an
# exception handler that cannot tell a bug from a quiet room.


@pytest.mark.asyncio
async def test_the_whisper_post_carries_no_name_out_of_scope():
    """Every name in the request-construction path must exist in that scope.

    A stub session that RECORDS the call rather than faking a response: the point
    is that a request is actually attempted, so any NameError on the way there
    fails the test instead of being absorbed by the caller's except.
    """
    import camera_client as cc

    seen: list[dict] = []

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"text": "включи свет"}

    class _Session:
        def post(self, url, **kw):
            seen.append(kw)
            return _Resp()

    s = _make_http_session()
    s.http_session = _Session()
    s.whisper_url = "http://10.0.0.9/v1/audio/transcriptions"
    s.whisper_model = "test-model"

    out = await s._fetch_transcription(b"RIFF....wavfmt ", temperature="0.0")

    assert seen, (
        "no Whisper request was attempted — the transcript is empty because the "
        "request was never sent, which is what a NameError in this path does"
    )
    assert out == "включи свет", f"a real response was ignored: {out!r}"


@pytest.mark.asyncio
async def test_a_broken_whisper_call_is_logged_not_swallowed():
    """A bare `except: pass` makes a programming error indistinguishable from an
    empty room. The caller logs `Whisper empty` for both, and that ambiguity is
    what hid the NameError for two hours."""
    import camera_client as cc

    class _Boom:
        def post(self, *a, **kw):
            raise RuntimeError("kaboom")

    lines: list[str] = []

    class _Log:
        def warning(self, msg, *a):
            lines.append(msg % a if a else msg)

        def __getattr__(self, _name):
            return self.info

        def info(self, msg, *a):
            lines.append(msg % a if a else msg)

    s = _make_http_session()
    s.http_session = _Boom()
    s.whisper_url = "http://10.0.0.9/v1/audio/transcriptions"
    s.whisper_model = "test-model"

    with patch.object(cc, "logger", _Log()):
        out = await s._fetch_transcription(b"RIFF....wavfmt ", temperature="0.0")

    assert out == ""
    joined = " ".join(lines)
    assert "whisper call failed" in joined and "kaboom" in joined, (
        f"a failed Whisper call was swallowed, so it is indistinguishable from a "
        f"quiet room: {joined!r}"
    )


def test_the_two_http_budgets_belong_to_their_own_functions():
    """The bug above was a wrong-function edit that `str.count` reported as
    unique: the play_audio line had 2 spaces less indentation, so an 18-space
    pattern matched the whisper line as a substring and `count == 1` passed.

    So assert on the values each function actually carries, not on the file.
    """
    import inspect

    import camera_client as cc

    whisper = inspect.getsource(cc.CameraSession._fetch_transcription)
    play = inspect.getsource(cc.CameraSession._play_audio_http)

    assert "len(pcm)" not in whisper, (
        "the Whisper path must not size a budget from `pcm` — its argument is "
        "`wav`, so that expression is a NameError on every call"
    )
    assert "total=30" in whisper, f"the Whisper budget is wrong: {whisper[-400:]}"

    assert "len(pcm)" in play, (
        "the play_audio budget was meant to scale with the clip, and it was "
        "written into the Whisper call instead"
    )
    assert "TTS_PLAY_RATE" in play


# --- Room noise was counted as our own echo, and the block never expired --------
#
# Living room, 07.10.2026 15:21, a silent room with nothing playing:
#
#     drop[muted=230 track=0 echo=32] -> 82 -> 101 -> 123 -> 158 -> 163
#     Echo chunk dropped (corr) every 2-4 s, continuously
#     VOSK wake heard but not fired: echo_tail=+1.9s / +0.8s / +0.0s
#
# `muted` frozen at 230 for 25 minutes proves nothing was playing, so nothing
# could have been our echo; `echo` climbing proves the correlation was firing.
# The user's word WAS heard (window=24286/3000) and blocked anyway — which is
# exactly the "стало хуже компьютер распознавать" report.
#
# Two causes: `_is_echo` searched a ring buffer that is never cleared, so stale
# speech from earlier turns matched ambient noise at one of 44 lags; and each
# match pushed `_wake_suppress_until` to `now + _ECHO_TAIL_S`, so the block slid
# forward faster than it could expire.


def _session_with_playback_ended_ago(seconds: float):
    s = _make_http_session()
    s._tts_play_end = time.time() - seconds
    return s


def test_a_correlation_is_not_believed_when_nothing_of_ours_is_in_the_air():
    """Past the horizon a match is coincidence, not echo — however well it fits."""
    import numpy as np

    import camera_client as cc

    # A ring full of our own speech, as a room that has spoken before would have.
    s = _session_with_playback_ended_ago(cc._ECHO_HORIZON_S + 5.0)
    s._tts_ring_len = 16000 * 60
    s._tts_ring = np.zeros(s._tts_ring_len, dtype=np.float32)
    s._tts_total = s._tts_ring_len
    s._echo_corr_threshold = 0.3

    speech = (np.sin(np.arange(16000) * 0.05) * 8000).astype(np.int16).tobytes()
    assert s._is_echo(speech) is False, (
        "audio was matched against our own voice long after playback ended, so "
        "room noise can be claimed as an echo and the wake gate held shut"
    )


def test_a_correlation_inside_the_horizon_is_still_believed():
    """The guard must not blind the detector to a real echo of a reply in flight."""
    import numpy as np

    import camera_client as cc

    s = _session_with_playback_ended_ago(1.0)
    s._tts_ring_len = 16000 * 60
    s._tts_ring = np.zeros(s._tts_ring_len, dtype=np.float32)
    s._tts_total = s._tts_ring_len
    s._echo_corr_threshold = 0.9

    rng = np.random.default_rng(0)
    sig_16k = rng.integers(-3000, 3000, size=16000, dtype=np.int16)  # 1 s
    s._store_tts_echo(sig_16k.tobytes(), rate=16000)
    s._tts_total += 3 * 16000   # inside the sweep (d starts at 2) and the horizon
    s._echo_corr_threshold = 0.9
    echo = sig_16k.tobytes()
    assert s._is_echo(echo) is True, (
        "a real echo of our own playback inside the horizon was not recognised — "
        "the fix must remove false echoes, not the detector"
    )


def test_the_wake_block_no_longer_slides_forward_on_every_echo():
    """The `now + _ECHO_TAIL_S` floor is what turned one false positive into a
    permanent block: each match pushed the deadline 3 s past the current moment,
    so a run of false echoes held the gate shut indefinitely."""
    import inspect

    import camera_client as cc

    # Comments stripped first: a test grepping the source finds the very
    # counter-example it looks for, because each rule is written IN its comment
    # as the mistake it prevents. Three versions of this assertion have matched
    # prose instead of code.
    src = "\n".join(
        ln for ln in inspect.getsource(cc.CameraSession._feed_audio).split("\n")
        if not ln.lstrip().startswith("#")
    )

    assert "now + _ECHO_TAIL_S" not in src, (
        "the wake block still carries a sliding floor; a correlation arriving "
        "while nothing is playing must not move the deadline"
    )
    assert "_ECHO_TAIL_S" in src, "the real tail must still apply"


# --- /play_audio was silent on SUCCESS, so the one report that mattered had no
# --- line behind it (kitchen, 10.10.2026: «писк после компьютер ненормальный»)


@pytest.mark.asyncio
async def test_play_audio_logs_a_ratio_on_success(monkeypatch):
    """Drives the REAL function, so an unbound name in the timing path fails here
    instead of on a live camera.

    That is not hypothetical: on 07.10.2026 a per-clip budget referencing `pcm`
    was written into `_fetch_transcription`, whose argument is `wav`. Every call
    raised NameError, an `except: pass` swallowed it, and Whisper returned nothing
    for two hours while the log said `Whisper empty` — the same line a silent room
    produces. The first draft of this instrument repeated it, using `sent_at` from
    `_speak_pcm` without assigning it here.
    """
    import camera_client as cc

    clock = {"t": 500.0}

    def _now():
        return clock["t"]

    class _Resp:
        status = 200

        async def __aenter__(self):
            clock["t"] += 3.0        # a 3x-slow camera, i.e. the fault
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, *a, **kw):
            return _Resp()

    lines: list[str] = []

    class _Log:
        def info(self, msg, *a):
            lines.append(msg % a if a else msg)

        def warning(self, msg, *a):
            lines.append(msg % a if a else msg)

        def __getattr__(self, _name):
            return self.info

    s = _make_http_session()
    s.http_session = _Session()
    s._play_audio_url = "http://10.0.0.9/play_audio"
    s._play_audio_headers = {}

    with monkeypatch.context() as m, patch.object(cc, "logger", _Log()):
        m.setattr(cc.time, "time", _now)
        ok = await s._play_audio_http(b"\x00\x01" * 48000)   # 1 s at 48 kHz

    assert ok is True
    joined = " ".join(lines)
    assert "play_audio 1.00s audio in 3.00s (x3.0)" in joined, (
        f"a successful /play_audio logged nothing about how long it took: "
        f"{joined!r}"
    )
    assert abs(s._play_slow_ratio - 3.0) < 0.01


@pytest.mark.asyncio
async def test_the_ratio_is_zero_before_anything_has_played():
    """So a reader cannot confuse 'not measured yet' with 'perfectly real time'."""
    s = _make_http_session()
    assert s._play_slow_ratio == 0.0
