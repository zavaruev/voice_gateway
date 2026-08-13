import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# Mock onnxruntime before importing engine to avoid errors since model file is missing
import onnxruntime as ort
ort.InferenceSession = MagicMock()

from engine import LocalAudioEngine

@pytest.fixture
def engine():
    engine = LocalAudioEngine()
    engine.oww_model = MagicMock()
    return engine

def test_check_wakeword_peak_gt_500(engine):
    """Test when peak > 500, it scales by peak."""
    audio = np.array([1000, -1000, 500, 0], dtype=np.int16)

    engine.oww_model.predict.return_value = {"model1": 0.5}

    result = engine.check_wakeword(audio, threshold=0.4)

    assert result is True
    # Verify predict was called with normalized array
    called_args = engine.oww_model.predict.call_args[0][0]
    expected = audio.astype(np.float32) / 1000.0
    np.testing.assert_array_equal(called_args, expected)

def test_check_wakeword_peak_le_500(engine):
    """Test when peak <= 500, it scales by 32768.0 * 32.0 and clips."""
    # Peak will be 100
    audio = np.array([100, -100, 50, 0], dtype=np.int16)

    engine.oww_model.predict.return_value = {"model1": 0.5}

    result = engine.check_wakeword(audio, threshold=0.4)

    assert result is True
    called_args = engine.oww_model.predict.call_args[0][0]

    # Calculate expected
    expected = audio.astype(np.float32) / 32768.0 * 32.0
    np.clip(expected, -1.0, 1.0, out=expected)
    np.testing.assert_array_equal(called_args, expected)

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
