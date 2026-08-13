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

    # Just after receive, time.time() - self._last_real_recv is very small
    # hold is min(15.0, 0.0 + 0.3) = 0.3s
    # So echo_active should be True
    assert track.echo_active() is True

    # Change _last_real_recv to simulate time passing
    track._last_real_recv = time.time() - 1.0
    assert track.echo_active() is False

    # Test with larger last_play_duration
    track._last_real_recv = time.time() - 2.0
    track._last_play_duration = 5.0
    # hold = min(15.0, 5.0 + 0.3) = 5.3s
    # 2.0 < 5.3, so should be True
    assert track.echo_active() is True

    track._last_real_recv = time.time() - 6.0
    # 6.0 < 5.3 is False
    assert track.echo_active() is False

def test_stop():
    track = AIVoiceOutputTrack(sample_rate=8000)
    # stop doesn't do anything currently, but we should test it can be called
    track.stop()
