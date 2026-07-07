import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Ensure clean environment
if "ADMIN_USERNAME" in os.environ:
    del os.environ["ADMIN_USERNAME"]
if "ADMIN_PASSWORD" in os.environ:
    del os.environ["ADMIN_PASSWORD"]

# Set environment variables for authentication and other dependencies
os.environ["NANOBOT_WS_URL"] = "ws://test_nanobot"
os.environ["WHISPER_URL"] = "http://test_whisper"
os.environ["TTS_URL"] = "http://test_tts"
os.environ["SPEAKER_ID_URL"] = "http://test_speaker"

with patch("main.ort.InferenceSession"):
    import main
    # Explicitly override the global vars since module might be already imported
    main.ADMIN_USERNAME = "admin"
    main.ADMIN_PASSWORD = "password"
    app = main.app

client = TestClient(app)

def test_ota_auth_required():
    response = client.get("/ota")
    assert response.status_code == 401

    response = client.post("/ota")
    assert response.status_code == 401

def test_ota_auth_success():
    response = client.get("/ota", auth=("admin", "password"))
    assert response.status_code == 200

    response = client.post("/ota", auth=("admin", "password"))
    assert response.status_code == 200

def test_ota_auth_failure():
    response = client.get("/ota", auth=("admin", "wrongpassword"))
    assert response.status_code == 401
