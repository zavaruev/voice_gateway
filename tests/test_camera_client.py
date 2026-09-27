"""Tests for camera_client.AIVoiceOutputTrack (av-based audio output track).

Covers the PyAV track wrapper the camera stream uses to play TTS audio:
initialization, packet timestamps and frame assembly. NOTE: this module
imports `av`, which is only installed inside the Docker image — on a host
without av the whole `pytest tests/` run fails at collection, so run the
5-file suite (see AGENTS/README) instead of the full directory.
"""
import asyncio
import time
import pytest
import av
import fractions

from camera_client import AIVoiceOutputTrack

@pytest.mark.asyncio
async def test_aivoiceoutputtrack_initialization():
    track = AIVoiceOutputTrack(sample_rate=16000)
    assert track.kind == "audio"
    assert track._sample_rate == 16000
    assert track._queue.maxsize == 500
    assert track._frame_count == 0

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


@pytest.mark.asyncio
async def test_is_echo_detects_own_tts():
    s = _make_session()
    rng = np.random.default_rng(0)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    sig_16k = np.repeat(sig_8k, 2)  # what _store_tts_echo upsamples to
    s._store_tts_echo(sig_8k.tobytes())
    # total is now 2560; simulate 10s passing (echo returns later)
    s._tts_total += 10 * 16000
    # The echoed chunk (same content) must be detected as echo
    assert s._is_echo(sig_16k.tobytes()) is True


@pytest.mark.asyncio
async def test_is_echo_rejects_other_speech():
    s = _make_session()
    rng = np.random.default_rng(1)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    other_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    other_16k = np.repeat(other_8k, 2)
    s._store_tts_echo(sig_8k.tobytes())
    s._tts_total += 10 * 16000
    # Different speech must NOT be flagged as echo
    assert s._is_echo(other_16k.tobytes()) is False


@pytest.mark.asyncio
async def test_is_echo_rejects_silence():
    s = _make_session()
    rng = np.random.default_rng(2)
    sig_8k = (rng.integers(-3000, 3000, size=1280, dtype=np.int16))
    silence = np.zeros(2560, dtype=np.int16)
    s._store_tts_echo(sig_8k.tobytes())
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
