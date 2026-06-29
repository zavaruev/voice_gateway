import pytest
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock onnxruntime before importing main
import onnxruntime as ort
ort.InferenceSession = MagicMock()

import main
from main import denoise_audio

def test_denoise_audio_happy_path():
    """Test that denoise_audio works correctly with valid PCM16 data."""
    # Create dummy 1-sec PCM16 buffer
    audio_data = (np.random.randn(16000) * 1000).astype(np.int16).tobytes()

    # We must mock df to not be None to trigger the actual logic
    with patch("main.df", MagicMock(), create=True):
        # Run denoise
        denoised_data = denoise_audio(audio_data, sample_rate=16000)

        # It should return bytes of the same length
        assert isinstance(denoised_data, bytes)
        assert len(denoised_data) == len(audio_data)

def test_denoise_audio_df_none():
    """Test that denoise_audio returns the original data if df is None."""
    audio_data = (np.random.randn(16000) * 1000).astype(np.int16).tobytes()

    # Run denoise with df not mocked (it will be missing or None)
    with patch("main.df", None, create=True):
        denoised_data = denoise_audio(audio_data, sample_rate=16000)

    assert denoised_data == audio_data

def test_denoise_audio_reduce_noise_exception():
    """Test that denoise_audio returns the original data if noisereduce fails."""
    audio_data = (np.random.randn(16000) * 1000).astype(np.int16).tobytes()

    with patch("main.df", MagicMock(), create=True):
        with patch("main.nr.reduce_noise", side_effect=Exception("mocked reduce_noise error")):
            denoised_fail = denoise_audio(audio_data, sample_rate=16000)

            # Should return the original data upon failure
            assert denoised_fail == audio_data

def test_denoise_audio_invalid_input():
    """Test that denoise_audio handles invalid input gracefully and returns original."""
    class BadBuffer:
        def __len__(self):
            return 10

    bad_input = BadBuffer()
    with patch("main.df", MagicMock(), create=True):
        # It should fail at np.frombuffer and return the original bad_input
        result = denoise_audio(bad_input, sample_rate=16000)

    assert result == bad_input

def test_denoise_audio_missing_module():
    """Test that denoise_audio handles the module being unavailable gracefully."""
    audio_data = (np.random.randn(16000) * 1000).astype(np.int16).tobytes()

    with patch("main.df", MagicMock(), create=True):
        # We can simulate the module being missing by mocking main.nr to None
        with patch("main.nr", None):
            result = denoise_audio(audio_data, sample_rate=16000)
            # It should trigger an AttributeError and return the original data
            assert result == audio_data
