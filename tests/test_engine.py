import pytest
import asyncio
import av
import fractions
from engine import AudioStreamTrack

@pytest.mark.asyncio
async def test_audio_stream_track_init():
    track = AudioStreamTrack()
    assert track.kind == "audio"
    assert track._pts == 0
    assert isinstance(track._queue, asyncio.Queue)

@pytest.mark.asyncio
async def test_audio_stream_track_put_and_recv():
    track = AudioStreamTrack()

    # Create a dummy frame
    frame = av.AudioFrame(format='s16', layout='mono', samples=960)

    # Put frame
    await track.put_frame(frame)

    # Receive frame
    received_frame = await track.recv()

    assert received_frame is frame

@pytest.mark.asyncio
async def test_audio_stream_track_recv_timeout():
    track = AudioStreamTrack()

    # Receive frame without putting any
    # should timeout and return a dummy frame with zeros
    received_frame = await track.recv()

    assert isinstance(received_frame, av.AudioFrame)
    assert received_frame.format.name == 's16'
    assert received_frame.layout.name == 'mono'
    assert received_frame.samples == 960
    assert received_frame.sample_rate == 48000
    assert received_frame.pts == 0
    assert received_frame.time_base == fractions.Fraction(1, 48000)

    # Check that planes[0] contains zeros
    assert bytes(received_frame.planes[0]) == b'\x00' * 1920

    # Check that pts was updated
    assert track._pts == 960

    # Receive again to check pts update
    received_frame_2 = await track.recv()
    assert received_frame_2.pts == 960
    assert track._pts == 1920
