import sys
from unittest.mock import MagicMock
import numpy as np
import pytest
import asyncio
import av
import fractions

# Mock onnxruntime before importing engine to avoid errors since model file is missing
import onnxruntime as ort
ort.InferenceSession = MagicMock()

from engine import LocalAudioEngine, AudioStreamTrack

@pytest.fixture
def engine():
    engine = LocalAudioEngine()
    engine.oww_model = MagicMock()
    return engine

def test_check_wakeword_predict_raw_int16(engine):
    """Test predict is called with raw int16 samples (openwakeword pipeline
    casts its buffer with .astype(np.int16); normalized floats would be
    quantized to {-1,0,1})."""
    audio = np.array([1000, -1000, 500, 0], dtype=np.int16)

    engine.oww_model.predict.return_value = {"model1": 0.5}

    result = engine.check_wakeword(audio, threshold=0.4)

    assert result is True
    called_args = engine.oww_model.predict.call_args[0][0]
    np.testing.assert_array_equal(called_args, audio)
    assert called_args.dtype == np.int16

def test_check_wakeword_prediction_empty(engine):
    """Test when prediction is empty, score is 0.0 and returns False."""
    audio = np.array([100, -100], dtype=np.int16)

    # Return empty dict
    engine.oww_model.predict.return_value = {}

    result = engine.check_wakeword(audio, threshold=0.4)
    assert result is False

def test_check_wakeword_score_le_threshold(engine):
    """Test when score <= threshold, returns False."""
    audio = np.array([1000, -1000], dtype=np.int16)

    # Score is exactly threshold
    engine.oww_model.predict.return_value = {"model1": 0.4}

    result = engine.check_wakeword(audio, threshold=0.4)
    assert result is False

def test_check_wakeword_logging_condition(engine, caplog):
    """Test the _ww_calls % 500 == 0 branch and RMS calculation."""
    audio = np.array([100, -100, 100, -100], dtype=np.int16)

    engine.oww_model.predict.return_value = {"model1": 0.2}
    engine._ww_calls = 500 # Set to trigger modulo 500

    with caplog.at_level("INFO"):
        engine.check_wakeword(audio, threshold=0.4)

    assert "WW peek" in caplog.text
    assert "rms=" in caplog.text
    assert engine._ww_calls == 501 # ensure it increments

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
