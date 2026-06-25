import sys
from unittest.mock import MagicMock
import time

# Mock onnxruntime before importing main
sys.modules['onnxruntime'] = MagicMock()

import main
import pytest

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the chat id cache before each test."""
    main.CHAT_ID_CACHE.clear()

def test_get_cached_chat_id_valid():
    mac = "00:11:22:33:44:55"
    chat_id = "test-chat-id-123"

    # Set the cache
    main.CHAT_ID_CACHE[mac.lower()] = {"chat_id": chat_id, "ts": time.time()}

    assert main.get_cached_chat_id(mac) == chat_id

def test_get_cached_chat_id_expired():
    mac = "AA:BB:CC:DD:EE:FF"
    chat_id = "test-chat-id-expired"

    # Set the timestamp in the past to exceed CHAT_ID_TTL
    main.CHAT_ID_CACHE[mac.lower()] = {"chat_id": chat_id, "ts": time.time() - main.CHAT_ID_TTL - 100}

    assert main.get_cached_chat_id(mac) is None

def test_get_cached_chat_id_uncached():
    mac = "12:34:56:78:90:AB"

    # Ensure the MAC is not in cache
    main.CHAT_ID_CACHE.pop(mac.lower(), None)

    assert main.get_cached_chat_id(mac) is None

def test_get_cached_chat_id_case_insensitive():
    mac_upper = "FF:EE:DD:CC:BB:AA"
    mac_lower = mac_upper.lower()
    chat_id = "test-chat-id-case"

    # Cache it with lower case key as the original implementation expects
    main.CHAT_ID_CACHE[mac_lower] = {"chat_id": chat_id, "ts": time.time()}

    # Should work for uppercase input
    assert main.get_cached_chat_id(mac_upper) == chat_id
    # Should work for lowercase input
    assert main.get_cached_chat_id(mac_lower) == chat_id
