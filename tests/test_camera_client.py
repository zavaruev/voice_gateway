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
import av
import fractions

from camera_client import (
    AIVoiceOutputTrack,
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
    # total is now 2560; simulate 10s passing (echo returns later)
    s._tts_total += 10 * 16000
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
    s._tts_total += 10 * 16000
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
    s._tts_total += 10 * 16000
    # Different speech must NOT be flagged as echo
    assert s._is_echo(other_16k.tobytes()) is False


@pytest.mark.asyncio
async def test_is_echo_rejects_silence():
    s = _make_session()
    rng = np.random.default_rng(2)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    silence = np.zeros(2560, dtype=np.int16)
    s._store_tts_echo(sig_8k.tobytes(), rate=8000)
    s._tts_total += 10 * 16000
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
    being collected."""
    s = _gate_session()
    s._last_wake_fired_at = 1000.0
    assert s._wake_gate_open(1000.8) is False
    assert "same_breath" in s._wake_gate_reason(1000.8)
    assert s._wake_gate_open(1002.0) is True


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

