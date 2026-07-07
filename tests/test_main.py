import asyncio
import pytest
import os
import json
import time
import numpy as np
import struct
from unittest.mock import patch, MagicMock, mock_open

# Environment variable mocks needed before importing main
os.environ["NANOBOT_WS_URL"] = "ws://test_nanobot"
os.environ["WHISPER_URL"] = "http://test_whisper"
os.environ["TTS_URL"] = "http://test_tts"
os.environ["SPEAKER_ID_URL"] = "http://test_speaker"

from main import (
    is_valid_text,
    make_chat_id,
    calculate_rms,
    pack_ogg,
    set_cached_chat_id,
    get_cached_chat_id,
    load_chat_id_cache,
    save_chat_id_cache,
    CHAT_ID_TTL,
    clear_expired_chat_ids,
    VadEngine,
    WHISPER_HALLUCINATIONS,
    SINGLE_WORD_HALLUCINATIONS,
    normalize_mac,
)
import main

def test_is_valid_text():
    # Test valid texts
    assert is_valid_text("Привет, как дела?") == True
    assert is_valid_text("Это нормальный текст для проверки.") == True

    # Test single word hallucinations
    for word in SINGLE_WORD_HALLUCINATIONS:
        assert is_valid_text(word) == False

    # Test whisper hallucinations
    for bad_phrase in WHISPER_HALLUCINATIONS:
        assert is_valid_text(f"Ой, {bad_phrase} что-то там") == False

    # Test short single words
    assert is_valid_text("да") == False
    assert is_valid_text("нет") == True # "нет" length is 3, valid?

    # Test empty or punctuation only
    assert is_valid_text("") == False
    assert is_valid_text(" . , ?! - ") == False

    # Test repeated characters
    assert is_valid_text("ааааааааааааааааааа") == False

def test_normalize_mac():
    # Standard uppercase MAC
    assert normalize_mac("AA:BB:CC:DD:EE:FF") == "AA:BB:CC:DD:EE:FF"

    # Lowercase MAC
    assert normalize_mac("aa:bb:cc:dd:ee:ff") == "AA:BB:CC:DD:EE:FF"

    # MAC with leading/trailing whitespaces
    assert normalize_mac("  aa:bb:cc:dd:ee:ff \n\t") == "AA:BB:CC:DD:EE:FF"

    # Mixed case MAC
    assert normalize_mac("aA:bB:Cc:DD:ee:Ff") == "AA:BB:CC:DD:EE:FF"

    # Empty string
    assert normalize_mac("") == ""

    # Whitespace-only string
    assert normalize_mac("   \n\t  ") == ""


def test_make_chat_id():
    mac1 = "AA:BB:CC:DD:EE:FF"
    mac2 = "aa:bb:cc:dd:ee:ff"
    mac3 = "11:22:33:44:55:66"

    # Same MAC different case should yield same ID
    assert make_chat_id(mac1) == make_chat_id(mac2)
    # Different MAC should yield different ID
    assert make_chat_id(mac1) != make_chat_id(mac3)
    # Ensure correct format (uuid)
    chat_id = make_chat_id(mac1)
    assert len(chat_id) == 36
    assert chat_id.count("-") == 4


def test_calculate_rms():
    # Create an empty audio buffer
    empty_audio = b""
    assert calculate_rms(empty_audio) == 0.0

    # Create a 1kHz sine wave test audio at 16000Hz (16-bit PCM)
    t = np.linspace(0, 1, 16000, False)
    sine_wave = np.sin(2 * np.pi * 1000 * t) * 32767
    sine_wave_int16 = sine_wave.astype(np.int16)
    rms = calculate_rms(sine_wave_int16.tobytes())

    # RMS of a sine wave with amplitude 1 is ~0.707.
    # Our float values are scaled down by 32768, so the expected RMS is ~0.707
    assert 0.70 < rms < 0.71

def test_pack_ogg():
    # Test packing 1 Opus frame into Ogg container
    sample_rate = 16000
    dummy_frame = b"dummy_opus_data"

    # 2 frames
    frames = [dummy_frame, dummy_frame]
    ogg_data = pack_ogg(frames, sample_rate)

    # Check that it creates some bytes output containing OggS header
    assert isinstance(ogg_data, bytes)
    assert len(ogg_data) > 0
    assert ogg_data.startswith(b'OggS')

    # It should contain OpusHead and OpusTags headers
    assert b'OpusHead' in ogg_data
    assert b'OpusTags' in ogg_data

def test_pack_ogg_empty_frames():
    sample_rate = 16000
    ogg_data = pack_ogg([], sample_rate)

    assert isinstance(ogg_data, bytes)
    assert len(ogg_data) > 0
    assert ogg_data.startswith(b'OggS')

    # Should contain headers but no data pages (since no frames)
    assert b'OpusHead' in ogg_data
    assert b'OpusTags' in ogg_data

def test_pack_ogg_pagination():
    sample_rate = 16000
    dummy_frame = b"dummy"

    # Test with exactly 50 frames
    frames_50 = [dummy_frame] * 50
    ogg_data_50 = pack_ogg(frames_50, sample_rate)

    # Find all OggS headers to count pages
    # First page: OpusHead (bos), Second: OpusTags, Third: Audio data (eos)
    assert ogg_data_50.count(b'OggS') == 3

    # Test with 101 frames to verify it splits into 50, 50, 1
    frames_101 = [dummy_frame] * 101
    ogg_data_101 = pack_ogg(frames_101, sample_rate)

    # OpusHead, OpusTags, Data(50), Data(50), Data(1) -> 5 pages total
    assert ogg_data_101.count(b'OggS') == 5

    # To check that correct flags are applied:
    # First byte after OggS, Version=0, Flags byte (5th byte from 'OggS')
    # OggS(4), Version(1), HeaderType(1) -> offset is 5 from OggS index

    idx = ogg_data_101.find(b'OggS')
    page_count = 0
    while idx != -1:
        flag = ogg_data_101[idx + 5]
        if page_count == 0:
            assert flag == 0x02 # BOS
        elif page_count == 4:
            assert flag == 0x04 # EOS
        else:
            assert flag == 0x00 # Normal page

        page_count += 1
        idx = ogg_data_101.find(b'OggS', idx + 1)

@patch("main.time.time")
def test_chat_id_cache(mock_time):
    # Setup
    mock_time.return_value = 1000.0
    main.CHAT_ID_CACHE.clear()

    mac = "aa:bb:cc:dd:ee:ff"
    chat_id = "test-chat-id-123"

    # Mock file I/O for saving cache
    with patch("main.open", mock_open()) as m_open:
        set_cached_chat_id(mac, chat_id)

        # Verify it was cached in memory
        assert main.CHAT_ID_CACHE[mac] == {"chat_id": chat_id, "ts": 1000.0}

        # Verify it was saved to disk
        m_open.assert_called_with(main.CHAT_ID_CACHE_FILE, "w")

    # Get cached ID within TTL
    mock_time.return_value = 1000.0 + (CHAT_ID_TTL - 100)
    assert get_cached_chat_id(mac) == chat_id

    # Get cached ID after TTL
    mock_time.return_value = 1000.0 + (CHAT_ID_TTL + 100)
    assert get_cached_chat_id(mac) is None

    # Clear expired
    main.CHAT_ID_CACHE[mac] = {"chat_id": chat_id, "ts": 1000.0}
    with patch("main.open", mock_open()):
        clear_expired_chat_ids()
        # Should be removed because time is 1000 + TTL + 100
        assert mac not in main.CHAT_ID_CACHE

def test_load_chat_id_cache():
    main.CHAT_ID_CACHE.clear()

    # Create valid dummy cache data
    valid_data = {
        "aa:bb:cc:dd:ee:ff": {"chat_id": "id1", "ts": time.time()}
    }

    with patch("os.path.exists", return_value=True):
        with patch("main.open", mock_open(read_data=json.dumps(valid_data))):
            load_chat_id_cache()

            assert "aa:bb:cc:dd:ee:ff" in main.CHAT_ID_CACHE
            assert main.CHAT_ID_CACHE["aa:bb:cc:dd:ee:ff"]["chat_id"] == "id1"

@patch("main.ort.InferenceSession")
def test_vad_engine(mock_inference_session):
    # We mock the InferenceSession so we don't really load ONNX in this test
    # but still test the logic of VadEngine.

    # Mocking the run output to simulate "speech detected"
    # out = [[[0.5]]] where > threshold (0.15)
    mock_session_instance = MagicMock()
    mock_session_instance.run.return_value = (np.array([[[0.5]]]), np.zeros((2, 1, 128), dtype=np.float32))
    mock_inference_session.return_value = mock_session_instance

    vad = VadEngine()
    assert vad.noise_floor == 0.02
    assert vad.threshold == 0.15

    # Provide 512+ samples of audio
    audio_data = (np.ones(600) * 1000).astype(np.int16).tobytes()
    is_speech, rms = asyncio.run(vad.is_speech(audio_data))

    assert is_speech == True
    # The session should have been called since we provided enough buffer
    assert mock_session_instance.run.called

    # Reset should clear state
    vad.reset()
    assert len(vad.buffer) == 0

    # Provide small audio -> no run should happen, returns False unless RMS is > 0.02
    # With np.ones(10)*10, RMS is ~0.0003, no speech
    mock_session_instance.run.reset_mock()
    small_audio_data = (np.ones(10) * 10).astype(np.int16).tobytes()
    is_speech, rms = asyncio.run(vad.is_speech(small_audio_data))
    assert not mock_session_instance.run.called
    assert is_speech == False
