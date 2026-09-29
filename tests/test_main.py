"""Tests for main.py helpers (chat-id cache, RMS, text/sentence utils, routes).

IMPORTANT: this module mutates os.environ BEFORE importing main — main.py
reads every config value at import time, so the test env must exist first.
It needs numpy/aiohttp (and, through main, opuslib) — available in Docker;
on the host this file fails at collection, use the 5-file suite instead.
"""
import pytest
import asyncio
import os
import json
import time
import numpy as np
import aiohttp
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

# Environment variable mocks needed before importing main
os.environ["NANOBOT_WS_URL"] = "ws://test_nanobot"
os.environ["WHISPER_URL"] = "http://test_whisper"
os.environ["TTS_URL"] = "http://test_tts"
os.environ["SPEAKER_ID_URL"] = "http://test_speaker"

from main import (
    fetch_transcription,
    is_valid_text,
    make_chat_id,
    calculate_rms,
    pack_ogg,
    set_cached_chat_id,
    get_cached_chat_id,
    verify_auth,
    load_chat_id_cache,
    CHAT_ID_TTL,
    VadEngine,
    load_speaker_names,
    load_firmware_meta,
    save_firmware_meta,
    normalize_mac,
    device_online_status,
    create_tracked_task,
)
from audio_utils import (
    WHISPER_HALLUCINATIONS,
    SINGLE_WORD_HALLUCINATIONS,
)
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
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

def test_device_online_status():
    dummy_states = {
        "client1": {"mac": "AA:BB:CC:11:22:33", "status": "online"},
        "client2": {"mac": "aa:bb:cc:dd:ee:ff", "status": "playing"},
        "client3": {"status": "recording"}, # No MAC
    }

    with patch.dict(main.session_states, dummy_states, clear=True):
        # Test exact match
        assert device_online_status("AA:BB:CC:11:22:33") == "online"

        # Test case-insensitive match (search with lower, stored is upper)
        assert device_online_status("aa:bb:cc:11:22:33") == "online"

        # Test case-insensitive match (search with upper, stored is lower)
        assert device_online_status("AA:BB:CC:DD:EE:FF") == "playing"

        # Test non-existent MAC
        assert device_online_status("00:11:22:33:44:55") == "offline"

        # Test empty string MAC
        assert device_online_status("") == "offline"

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


@patch.dict(main.__dict__, {"NANOBOT_SESSION_SALT": ""})
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

    with patch.dict(main.__dict__, {"NANOBOT_SESSION_SALT": "my_secret_salt"}):
        # Different salt changes ID
        assert make_chat_id(mac1) != chat_id
        assert make_chat_id(mac1) == make_chat_id(mac2)


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
def test_get_cached_chat_id(mock_time):
    mock_time.return_value = 1000.0
    mac = "aa:bb:cc:dd:ee:ff"
    chat_id = "test-chat-id-123"

    test_cache = {
        mac: {"chat_id": chat_id, "ts": 1000.0}
    }

    with patch.dict(main.CHAT_ID_CACHE, test_cache, clear=True):
        # 1. Happy path (within TTL)
        assert get_cached_chat_id(mac) == chat_id

        # 2. Case insensitivity
        assert get_cached_chat_id(mac.upper()) == chat_id

        # 3. Missing key
        assert get_cached_chat_id("00:11:22:33:44:55") is None

        # 4. TTL expiration
        mock_time.return_value = 1000.0 + main.CHAT_ID_TTL + 1.0
        assert get_cached_chat_id(mac) is None


@patch("main.time.time")
def test_chat_id_cache(mock_time):
    # Setup
    mock_time.return_value = 1000.0
    main.CHAT_ID_CACHE.clear()

    mac = "aa:bb:cc:dd:ee:ff"
    chat_id = "test-chat-id-123"

    # We only test retrieval logic here; setting logic is heavily tested separately
    main.CHAT_ID_CACHE[mac] = {"chat_id": chat_id, "ts": 1000.0}

    # Get cached ID within TTL
    mock_time.return_value = 1000.0 + (CHAT_ID_TTL - 100)
    assert get_cached_chat_id(mac) == chat_id

    # Get cached ID after TTL
    mock_time.return_value = 1000.0 + (CHAT_ID_TTL + 100)
    assert get_cached_chat_id(mac) is None


def test_set_cached_chat_id_with_loop():
    main.CHAT_ID_CACHE.clear()
    with patch.dict(main.__dict__, {"_chat_id_last_save": 0.0}):
        with patch("main.time.time") as mock_time:
            mock_time.return_value = 2000.0
            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop

                set_cached_chat_id("AA:BB:CC", "chat1")

                assert main.CHAT_ID_CACHE["aa:bb:cc"] == {"chat_id": "chat1", "ts": 2000.0}
                mock_loop.run_in_executor.assert_called_once_with(None, main.save_chat_id_cache, main.CHAT_ID_CACHE)

                # Test throttling
                mock_loop.run_in_executor.reset_mock()
                mock_time.return_value = 2004.0
                set_cached_chat_id("AA:BB:CC", "chat2")

                assert main.CHAT_ID_CACHE["aa:bb:cc"] == {"chat_id": "chat2", "ts": 2004.0}
                mock_loop.run_in_executor.assert_not_called()


def test_set_cached_chat_id_no_loop():
    main.CHAT_ID_CACHE.clear()
    with patch.dict(main.__dict__, {"_chat_id_last_save": 0.0}):
        with patch("main.time.time") as mock_time:
            mock_time.return_value = 3000.0
            with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
                with patch("main.save_chat_id_cache") as mock_save:
                    set_cached_chat_id("DD:EE:FF", "chat3")

                    assert main.CHAT_ID_CACHE["dd:ee:ff"] == {"chat_id": "chat3", "ts": 3000.0}
                    mock_save.assert_called_once_with(main.CHAT_ID_CACHE)

                    # Test throttling
                    mock_save.reset_mock()
                    mock_time.return_value = 3004.0
                    set_cached_chat_id("DD:EE:FF", "chat4")

                    assert main.CHAT_ID_CACHE["dd:ee:ff"] == {"chat_id": "chat4", "ts": 3004.0}
                    mock_save.assert_not_called()

def test_load_speaker_names_success():
    valid_data = {"speaker_1": "Alice", "speaker_2": "Bob"}
    with patch("main.open", mock_open(read_data=json.dumps(valid_data))):
        result = load_speaker_names()
        assert result == valid_data

def test_load_speaker_names_error():
    with patch("main.open", side_effect=Exception("File read error")):
        result = load_speaker_names()
        assert result == {}

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
    assert vad.rms_noise_floor == 0.10
    assert vad.onnx_threshold == 0.02

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

def test_load_db_cache_hit():
    original_cache = main._DB_CACHE
    try:
        dummy_cache = {"device1": "config"}
        main._DB_CACHE = dummy_cache

        result = main.load_db()

        assert result == dummy_cache
        assert result is dummy_cache
    finally:
        main._DB_CACHE = original_cache

@patch("os.path.exists", return_value=True)
def test_load_db_from_file(mock_exists):
    original_cache = main._DB_CACHE
    try:
        main._DB_CACHE = None
        db_data = {"test_device": {"config": "val"}}
        with patch("builtins.open", mock_open(read_data=json.dumps(db_data))):
            result = main.load_db()
            assert result == db_data
            assert main._DB_CACHE == db_data
    finally:
        main._DB_CACHE = original_cache

@patch("os.path.exists", return_value=False)
def test_load_db_file_not_found(mock_exists):
    original_cache = main._DB_CACHE
    try:
        main._DB_CACHE = None
        result = main.load_db()
        assert result == {}
        assert main._DB_CACHE == {}
    finally:
        main._DB_CACHE = original_cache

@patch("os.path.exists", return_value=True)
def test_load_db_file_error(mock_exists):
    original_cache = main._DB_CACHE
    try:
        main._DB_CACHE = None
        # Simulate a JSON decoding error (e.g., malformed JSON)
        with patch("builtins.open", mock_open(read_data="{invalid_json}")):
            result = main.load_db()
            assert result == {}
            assert main._DB_CACHE == {}
    finally:
        main._DB_CACHE = original_cache


def test_load_firmware_meta_cache_hit():
    with patch.dict(main.__dict__, {"_firmware_meta_cache": {"version": "cached", "filename": "test.bin", "timestamp": 123}}):
        result = load_firmware_meta()
        assert result == {"version": "cached", "filename": "test.bin", "timestamp": 123}

def test_load_firmware_meta_success():
    with patch.dict(main.__dict__, {"_firmware_meta_cache": None}):
        with patch("builtins.open", mock_open(read_data='{"version": "1.0", "filename": "test.bin", "timestamp": 123}')):
            with patch("os.path.isfile", return_value=True):
                result = load_firmware_meta()
                assert result == {"version": "1.0", "filename": "test.bin", "timestamp": 123}
                assert main._firmware_meta_cache == {"version": "1.0", "filename": "test.bin", "timestamp": 123}

def test_load_firmware_meta_missing_bin():
    with patch.dict(main.__dict__, {"_firmware_meta_cache": None}):
        with patch("builtins.open", mock_open(read_data='{"version": "1.0", "filename": "missing.bin", "timestamp": 123}')):
            with patch("os.path.isfile", return_value=False):
                result = load_firmware_meta()
                assert result == {"version": "", "filename": "", "timestamp": 0}
                assert main._firmware_meta_cache == {"version": "", "filename": "", "timestamp": 0}

def test_load_firmware_meta_error():
    with patch.dict(main.__dict__, {"_firmware_meta_cache": None}):
        with patch("builtins.open", side_effect=Exception("JSON error")):
            result = load_firmware_meta()
            assert result == {"version": "", "filename": "", "timestamp": 0}
            assert main._firmware_meta_cache == {"version": "", "filename": "", "timestamp": 0}

@patch("main.time.time", return_value=1234567890.123)
def test_save_firmware_meta(mock_time):
    original_cache = main._firmware_meta_cache
    try:
        with patch("builtins.open", mock_open()) as m_open:
            result = save_firmware_meta("1.2.3", "firmware_v1.2.3.bin")

            expected_meta = {
                "version": "1.2.3",
                "filename": "firmware_v1.2.3.bin",
                "timestamp": int(1234567890.123 * 1000),  # 1234567890.123 * 1000
            }

            # verify return value
            assert result == expected_meta

            # verify global cache is updated
            assert main._firmware_meta_cache == expected_meta

            # verify file write
            m_open.assert_called_with(main.FIRMWARE_META, "w")

            # Get the file object that was written to
            handle = m_open()

            # Reconstruct what was written
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)

            # Load the JSON that was written and verify it matches expected
            written_json = json.loads(written_data)
            assert written_json == expected_meta
    finally:
        main._firmware_meta_cache = original_cache

def test_save_firmware_meta_error():
    original_cache = main._firmware_meta_cache
    try:
        with patch("builtins.open", side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Disk full"):
                save_firmware_meta("1.2.3", "firmware_v1.2.3.bin")

            # The cache shouldn't be updated if the file write fails
            assert main._firmware_meta_cache == original_cache
    finally:
        main._firmware_meta_cache = original_cache

def test_device_list_with_status():
    original_cache = main._DB_CACHE
    original_session = main.session_states.copy()
    try:
        # Set up a known db state
        db_data = {
            "AA:BB:CC:00:11:22": {"name": "Device 1"},
            "aa:bb:cc:00:11:33": {"name": "Device 2"}, # lowercase in DB
            "11:22:33:44:55:66": {"name": "Device 3"}
        }
        main._DB_CACHE = db_data

        # Scenario 1: All offline (session_states is empty)
        main.session_states.clear()
        result = main.device_list_with_status()
        assert len(result) == 3
        assert all(entry["status"] == "offline" for entry in result)
        # Check sorting: by MAC since all offline
        assert [entry["mac"] for entry in result] == ["11:22:33:44:55:66", "AA:BB:CC:00:11:22", "aa:bb:cc:00:11:33"]

        # Scenario 2: Partial online and MAC case insensitivity
        main.session_states.clear()
        main.session_states["ws1"] = {"mac": "aa:bb:cc:00:11:22", "status": "online"} # DB has uppercase, session has lowercase
        main.session_states["ws2"] = {"mac": "AA:BB:CC:00:11:33", "status": "online"} # DB has lowercase, session has uppercase

        result = main.device_list_with_status()
        assert len(result) == 3
        # Expected statuses
        status_map = {entry["mac"]: entry["status"] for entry in result}
        assert status_map["AA:BB:CC:00:11:22"] == "online"
        assert status_map["aa:bb:cc:00:11:33"] == "online"
        assert status_map["11:22:33:44:55:66"] == "offline"

        # Check sorting: online first, then by MAC
        assert result[0]["mac"] == "AA:BB:CC:00:11:22"
        assert result[1]["mac"] == "aa:bb:cc:00:11:33"
        assert result[2]["mac"] == "11:22:33:44:55:66"
    finally:
        main._DB_CACHE = original_cache
        main.session_states = original_session

@pytest.mark.asyncio
async def test_firmware_upload_path_traversal():
    from fastapi.testclient import TestClient
    from main import app, verify_auth

    # We will just patch FIRMWARE_DIR to avoid writing any files to disk
    patch("main.FIRMWARE_DIR", "/tmp/firmware_mock").start()
    patch("main.save_firmware_meta").start()
    patch("main.asyncio.to_thread").start() # Prevent actual file writing

    try:
        app.dependency_overrides[verify_auth] = lambda: "admin"
        client = TestClient(app)

        response = client.post(
            "/api/firmware/upload",
            data={"version": "$%&invalid"},
            files={"file": ("valid.bin", b"mock content")}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid version format"

        # 2. Test valid filename with version
        response = client.post(
            "/api/firmware/upload",
            data={"version": "1.0.0"},
            files={"file": ("valid.bin", b"mock content")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # 3. Test invalid filename (using file.filename)
        response = client.post(
            "/api/firmware/upload",
            files={"file": ("invalid$%&.bin", b"mock content")}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid filename"

        # 4. Test valid filename without version
        response = client.post(
            "/api/firmware/upload",
            files={"file": ("valid_name.bin", b"mock content")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    finally:
        # Clean up overrides and patches
        app.dependency_overrides = {}
        patch.stopall()

@pytest.mark.asyncio
async def test_fetch_transcription_success():
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"text": "Hello world"}
    mock_session.post.return_value.__aenter__.return_value = mock_response

    result = await fetch_transcription(b"fakeaudio", mock_session)
    assert result == "Hello world"
    mock_session.post.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_transcription_non_200():
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text.return_value = "Internal Server Error"
    mock_session.post.return_value.__aenter__.return_value = mock_response

    result = await fetch_transcription(b"fakeaudio", mock_session)
    assert result == ""
    mock_session.post.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_transcription_exception():
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    # Raising an exception when the context manager tries to enter
    mock_session.post.return_value.__aenter__.side_effect = aiohttp.ClientError("Network Error")

    result = await fetch_transcription(b"fakeaudio", mock_session)
    assert result == ""
    mock_session.post.assert_called_once()

from types import SimpleNamespace

def _auth_request():
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

def test_verify_auth_disabled():
    with patch("main.ADMIN_USERNAME", ""), patch("main.ADMIN_PASSWORD", ""):
        with pytest.raises(HTTPException) as exc:
            verify_auth(_auth_request(), HTTPBasicCredentials(username="admin", password="password"))
        assert exc.value.status_code == 401
        assert exc.value.detail == "Authentication disabled (credentials not configured)"

def test_verify_auth_no_credentials():
    with patch("main.ADMIN_USERNAME", "admin"), patch("main.ADMIN_PASSWORD", "password"):
        with pytest.raises(HTTPException) as exc:
            verify_auth(_auth_request(), None)
        assert exc.value.status_code == 401
        assert exc.value.detail == "Authentication required"

def test_verify_auth_incorrect_credentials():
    with patch("main.ADMIN_USERNAME", "admin"), patch("main.ADMIN_PASSWORD", "password"):
        with pytest.raises(HTTPException) as exc:
            verify_auth(_auth_request(), HTTPBasicCredentials(username="admin", password="wrongpassword"))
        assert exc.value.status_code == 401
        assert exc.value.detail == "Incorrect email or password"

def test_verify_auth_success():
    with patch("main.ADMIN_USERNAME", "admin"), patch("main.ADMIN_PASSWORD", "password"):
        result = verify_auth(_auth_request(), HTTPBasicCredentials(username="admin", password="password"))
        assert result == "admin"

@patch("main.json.dump")
@patch("builtins.open", new_callable=mock_open)
def test_save_chat_id_cache_success(mock_open_file, mock_json_dump):
    main.save_chat_id_cache({"some": "data"})
    mock_open_file.assert_called_once_with(main.CHAT_ID_CACHE_FILE, "w")
    mock_json_dump.assert_called_once_with({"some": "data"}, mock_open_file(), indent=2)

@patch("main.json.dump")
@patch("builtins.open", new_callable=mock_open)
def test_save_chat_id_cache_none(mock_open_file, mock_json_dump):
    main.CHAT_ID_CACHE = {"default": "val"}
    main.save_chat_id_cache(None)
    mock_open_file.assert_called_once_with(main.CHAT_ID_CACHE_FILE, "w")
    mock_json_dump.assert_called_once_with({"default": "val"}, mock_open_file(), indent=2)

@patch("builtins.open", side_effect=Exception("Test mock exception"))
@patch("main.logger.error")
def test_save_chat_id_cache_error(mock_logger_error, mock_open_err):
    main.save_chat_id_cache({"some": "data"})
    mock_logger_error.assert_called_once()
    assert "Failed to save chat_id cache:" in mock_logger_error.call_args[0][0]

@pytest.mark.asyncio
async def test_create_tracked_task():
    # Setup state
    state = {"tasks": set()}

    # Create a dummy coroutine
    async def dummy_coro():
        await asyncio.sleep(0.01)
        return "done"

    # Call the function
    task = create_tracked_task(dummy_coro(), state, name="test_task")

    # Verify task was added
    assert task in state["tasks"]
    assert task.get_name() == "test_task"

    # Await the task
    result = await task

    # We might need to yield to event loop for callback to fire
    await asyncio.sleep(0.01)

    # Verify task was removed
    assert result == "done"
    assert task not in state["tasks"]

@pytest.mark.asyncio
async def test_voice_ws_auth_bypass():
    with patch("main.NANOBOT_TOKEN", ""):
        from fastapi.testclient import TestClient
        from main import app

        # Test client requires auth overrides for the regular API but WS should hit voice_ws
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/?token=") as websocket:
                pass


# --- _hermes_player_task: the ack must not wait for sentence #2 ---------------
# Field case 29.09.2026, 11:39:26 -> 11:39:38: the backend enqueued the ack
# «Секунду, занимаюсь…» immediately, but the player awaited the NEXT queue
# item BEFORE playing the current one — so the ack stayed unsaid until L2's
# first sentence arrived. 12 s of dead air, i.e. exactly the silence the ack
# exists to cover (and it would be worse: an L2 run without an early sentence
# would sit unsaid for the whole turn).


@pytest.mark.asyncio
async def test_player_speaks_the_ack_without_the_next_sentence():
    q: asyncio.Queue = asyncio.Queue()
    played, synth = [], []

    async def fake_synth(text, state):
        synth.append(text)
        await asyncio.sleep(0.01)
        return text.encode()

    async def fake_stream(mp3, ws, sid, state, send_stop=True):
        played.append(mp3.decode())
        return True

    state = {"sid": "s1", "watchdog_fired": False}
    with patch.object(main, "synthesize_tts_mp3", fake_synth), patch.object(
        main, "stream_tts_pcm", fake_stream
    ):
        task = asyncio.create_task(main._hermes_player_task(q, None, state))
        await q.put("Секунду, занимаюсь…")
        # L2 is still working: the queue is EMPTY, yet the ack must play.
        await asyncio.sleep(0.2)
        assert played == ["Секунду, занимаюсь…"], (
            "the ack waited for the next queue item instead of playing"
        )
        await q.put("Пылесос сейчас на базе.")
        await q.put(None)  # backend end-of-stream marker
        await asyncio.wait_for(task, 2.0)

    # One-ahead pipeline preserved: every sentence synthesised exactly once,
    # in order, and the queue contents are what actually got spoken.
    assert synth == ["Секунду, занимаюсь…", "Пылесос сейчас на базе."]
    assert played == synth
    assert state["reply_sentences"] == [s + " " for s in synth]


@pytest.mark.asyncio
async def test_player_watchdog_fired_drains_the_queue_silently():
    """A fired watchdog means the apology already went out: the rest of the
    reply is drained, never spoken, and the prefetch task is not stranded."""
    q: asyncio.Queue = asyncio.Queue()
    played = []

    async def fake_synth(text, state):  # pragma: no cover - must not run
        raise AssertionError("no synthesis after the watchdog fired")

    async def fake_stream(mp3, ws, sid, state, send_stop=True):  # pragma: no cover
        played.append(mp3)
        return True

    state = {"sid": "s1", "watchdog_fired": True}
    with patch.object(main, "synthesize_tts_mp3", fake_synth), patch.object(
        main, "stream_tts_pcm", fake_stream
    ):
        await q.put("фраза из реплики")
        await q.put(None)
        await asyncio.wait_for(main._hermes_player_task(q, None, state), 2.0)

    assert played == []
    assert "reply_sentences" not in state
