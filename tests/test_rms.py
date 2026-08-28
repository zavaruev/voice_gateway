import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# Mock onnxruntime before importing main to avoid errors since model file is missing
import onnxruntime as ort
ort.InferenceSession = MagicMock()

from main import calculate_rms

def test_calculate_rms_empty():
    """Test calculate_rms with empty byte string."""
    assert calculate_rms(b'') == 0.0

def test_calculate_rms_less_than_two_bytes():
    """Test calculate_rms with a byte string that contains only 1 byte (less than 1 frame)."""
    assert calculate_rms(b'\x00') == 0.0

def test_calculate_rms_zeros():
    """Test calculate_rms with a byte string of zeros."""
    zeros = b'\x00' * 100
    assert calculate_rms(zeros) == 0.0

def test_calculate_rms_valid_data():
    """Test calculate_rms with valid PCM16 data."""
    # Create audio frame data with known RMS
    audio = np.array([16384, -16384, 0, 0], dtype=np.int16)
    # Normalized audio floats: 0.5, -0.5, 0, 0
    # Expected mean square = (0.25 + 0.25 + 0 + 0) / 4 = 0.125
    # Expected RMS = sqrt(0.125) ≈ 0.35355339
    expected_rms = np.sqrt(0.125)

    pcm_data = audio.tobytes()
    assert np.isclose(calculate_rms(pcm_data), expected_rms)

def test_calculate_rms_exception_handling():
    """Test that exceptions during calculation are caught and return 0.0."""
    # By passing an object that is not bytes but behaves enough like bytes to pass len,
    # or just something that fails np.frombuffer like a string
    class BadBuffer:
        def __len__(self):
            return 4

    assert calculate_rms(BadBuffer()) == 0.0
