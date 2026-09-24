import pytest
import asyncio
import os
import time
from unittest.mock import patch, MagicMock, AsyncMock

# Environment variable mocks needed before importing main
os.environ["NANOBOT_WS_URL"] = "ws://test_nanobot"
os.environ["WHISPER_URL"] = "http://test_whisper"
os.environ["TTS_URL"] = "http://test_tts"
os.environ["SPEAKER_ID_URL"] = "http://test_speaker"

with patch('main.ort.InferenceSession'):
    from main import process_audio_and_send
import camera_client


@pytest.fixture
def mock_state():
    return {
        "status": "PROCESSING",
        "sid": "test_sid",
        "http_session": MagicMock(),
        "tasks": set()
    }

@pytest.fixture
def mock_ws():
    return MagicMock()


@pytest.mark.asyncio
@patch("main.time.time")
async def test_process_audio_and_send_skips_when_tts_active(mock_time, mock_state, mock_ws):
    mock_time.return_value = 100.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    assert mock_state["status"] == "LISTENING"


@pytest.mark.asyncio
@patch("main.time.time")
async def test_process_audio_and_send_skips_few_frames(mock_time, mock_state, mock_ws):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0

    await process_audio_and_send([1]*10, mock_state, mock_ws)

    assert mock_state["status"] == "LISTENING"


@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
async def test_process_audio_and_send_skips_failed_metrics(mock_metrics, mock_time, mock_state, mock_ws):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.return_value = (False, 0.0, 0.0, None)

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    assert mock_state["status"] == "LISTENING"
    mock_metrics.assert_called_once()

@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
@patch("main.pack_ogg")
@patch("main.fetch_speaker_id", new_callable=AsyncMock)
@patch("main.fetch_transcription", new_callable=AsyncMock)
@patch("main._check_speaker_lock")
@patch("main.is_valid_text")
@patch("main._handle_successful_transcription", new_callable=AsyncMock)
@patch("main.trigger_emotion", new_callable=AsyncMock)
async def test_process_audio_and_send_success(
    mock_trigger_emotion, mock_handle_success, mock_is_valid, mock_check_lock,
    mock_fetch_stt, mock_fetch_uid, mock_pack,
    mock_metrics, mock_time, mock_state, mock_ws
):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.return_value = (True, 0.5, 0.8, None)
    mock_pack.return_value = b"ogg_data"

    mock_fetch_uid.return_value = "user_123"
    mock_fetch_stt.return_value = "hello world"
    mock_check_lock.return_value = True
    mock_is_valid.return_value = True

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    mock_pack.assert_called_once()
    mock_fetch_uid.assert_called_once_with(b"ogg_data", mock_state["http_session"])
    mock_fetch_stt.assert_called_once_with(b"ogg_data", mock_state["http_session"])
    mock_check_lock.assert_called_once_with("user_123", "test_sid")
    mock_is_valid.assert_called_once_with("hello world")
    mock_handle_success.assert_called_once_with("hello world", "user_123", mock_state, mock_ws, 120.0, 120.0)
    mock_trigger_emotion.assert_called_once_with("thinking", mock_ws, "test_sid")



@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
@patch("main.pack_ogg")
@patch("main.fetch_speaker_id", new_callable=AsyncMock)
@patch("main.fetch_transcription", new_callable=AsyncMock)
@patch("main._check_speaker_lock")
@patch("main.is_valid_text")
@patch("main._handle_rejected_transcription", new_callable=AsyncMock)
@patch("main.trigger_emotion", new_callable=AsyncMock)
async def test_process_audio_and_send_invalid_text_rejected(
    mock_trigger_emotion, mock_handle_rejected, mock_is_valid, mock_check_lock,
    mock_fetch_stt, mock_fetch_uid, mock_pack,
    mock_metrics, mock_time, mock_state, mock_ws
):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.return_value = (True, 0.5, 0.8, None)
    mock_pack.return_value = b"ogg_data"

    mock_fetch_uid.return_value = "user_123"
    mock_fetch_stt.return_value = "uhhh"
    mock_check_lock.return_value = True
    mock_is_valid.return_value = False

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    mock_is_valid.assert_called_once_with("uhhh")
    mock_handle_rejected.assert_called_once_with("uhhh", "user_123", mock_state, mock_ws, 0.5, 0.8, 20)

@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
@patch("main.pack_ogg")
@patch("main.fetch_speaker_id", new_callable=AsyncMock)
@patch("main.fetch_transcription", new_callable=AsyncMock)
@patch("main._check_speaker_lock")
@patch("main.trigger_emotion", new_callable=AsyncMock)
async def test_process_audio_and_send_speaker_lock_fails(
    mock_trigger_emotion, mock_check_lock,
    mock_fetch_stt, mock_fetch_uid, mock_pack,
    mock_metrics, mock_time, mock_state, mock_ws
):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.return_value = (True, 0.5, 0.8, None)
    mock_pack.return_value = b"ogg_data"

    mock_fetch_uid.return_value = "user_123"
    mock_fetch_stt.return_value = "hello world"
    mock_check_lock.return_value = False

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    mock_check_lock.assert_called_once_with("user_123", "test_sid")
    assert mock_state["status"] == "LISTENING"


@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
@patch("main.reset_to_standby", new_callable=AsyncMock)
async def test_process_audio_and_send_exception_handling(
    mock_reset, mock_metrics, mock_time, mock_state, mock_ws
):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.side_effect = Exception("Test Error")

    await process_audio_and_send([1]*20, mock_state, mock_ws)

    mock_reset.assert_called_once_with(mock_ws, mock_state)


@pytest.mark.asyncio
@patch("main.time.time")
@patch("main._process_audio_metrics_and_gates", new_callable=AsyncMock)
@patch("main.pack_ogg")
@patch("main.fetch_speaker_id", new_callable=AsyncMock)
@patch("main.fetch_transcription", new_callable=AsyncMock)
@patch("main._check_speaker_lock")
@patch("main.is_valid_text")
@patch("main._handle_successful_transcription", new_callable=AsyncMock)
@patch("main.trigger_emotion", new_callable=AsyncMock)
async def test_process_audio_and_send_creates_tasks(
    mock_trigger_emotion, mock_handle_success, mock_is_valid, mock_check_lock,
    mock_fetch_stt, mock_fetch_uid, mock_pack,
    mock_metrics, mock_time, mock_state, mock_ws
):
    mock_time.return_value = 120.0
    camera_client.GLOBAL_TTS_UNTIL = 110.0
    mock_metrics.return_value = (True, 0.5, 0.8, None)
    mock_pack.return_value = b"ogg_data"

    mock_fetch_uid.return_value = "user_123"
    mock_fetch_stt.return_value = "hello world"
    mock_check_lock.return_value = True
    mock_is_valid.return_value = True

    # Just to confirm the tasks get created, we can check that they finish successfully
    # since create_tracked_task effectively runs the coroutines and we await gathering them.
    await process_audio_and_send([1]*20, mock_state, mock_ws)

    # mock_state["tasks"] should be empty after gather and add_done_callback complete
    # Though add_done_callback might be scheduled, await asyncio.sleep(0) runs it
    await asyncio.sleep(0)
    assert len(mock_state["tasks"]) == 0
