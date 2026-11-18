"""Wiring tests for the provider chain: the field/env must actually reach the
call site. A chain class with no wiring is the 'parameter added but never
passed' bug this repository has hit repeatedly."""

import pytest

from camera_client import CameraConfig, CameraSession


@pytest.mark.asyncio
async def test_whisper_fallback_serves_the_turn(monkeypatch):
    cfg = CameraConfig(stream_name="cam", whisper_url="http://primary",
                       whisper_fallback_url="http://backup")
    s = CameraSession(cfg)
    calls = []

    async def fake_one(wav, temp, rms, url=None):
        calls.append(url)
        if not url or url.endswith("primary"):
            raise RuntimeError("connection refused")
        return "включи свет"

    monkeypatch.setattr(s, "_fetch_transcription_one", fake_one)
    assert await s._fetch_transcription(b"wav") == "включи свет"
    assert calls[-1] == "http://backup"


@pytest.mark.asyncio
async def test_no_whisper_fallback_keeps_the_old_single_call(monkeypatch):
    s = CameraSession(CameraConfig(stream_name="cam", whisper_url="http://primary"))
    calls = []

    async def fake_one(wav, temp, rms, url=None):
        calls.append(url)
        return "ок"

    monkeypatch.setattr(s, "_fetch_transcription_one", fake_one)
    assert await s._fetch_transcription(b"wav") == "ок"
    assert calls == [None]                # NOT the wrapper's explicit URL


@pytest.mark.asyncio
async def test_tts_fallback_is_read_from_env(monkeypatch):
    import main

    monkeypatch.setenv("TTS_FALLBACK_URL", "http://backup-tts")
    monkeypatch.setattr(main, "TTS_URL", "http://primary-tts")
    calls = []

    async def fake_url(text, state, speed, url=""):
        calls.append(url)
        return b"MP3" if url.endswith("backup-tts") else None

    monkeypatch.setattr(main, "_synthesize_tts_mp3_url", fake_url)
    assert await main.synthesize_tts_mp3("тест", {}) == b"MP3"
    assert calls == ["http://primary-tts", "http://backup-tts"]
