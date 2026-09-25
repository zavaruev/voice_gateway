"""Central env configuration for jev-router (cascade level 1)."""

import os


def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


# --- External services (all host-LAN addresses; service runs host-network) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.22.102:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.22.102:6333").rstrip("/")

HA_URL = os.getenv("HA_URL", "http://192.168.22.111:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")

OMNIROUTE_URL = os.getenv("OMNIROUTE_URL", "http://192.168.22.101:20128").rstrip("/")
OMNIROUTE_COMBO = os.getenv("OMNIROUTE_COMBO", "gemma4_31b_free")

HERMES_URL = os.getenv("HERMES_URL", "http://192.168.22.102:8642").rstrip("/")
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")

WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8092").rstrip("/")

# --- Routing thresholds (calibrated on qwen3-embedding:0.6b, Sep 2026) ---
# Positive same-intent paraphrases: 0.60..0.98 cosine; cross-route pairs: 0.34..0.61.
# Confidence = clip((score - 0.50) / 0.40, 0, 1) => conf 0.85 requires score >= 0.84.
CONFIDENCE_THRESHOLD = _f("CONFIDENCE_THRESHOLD", "0.85")
MARGIN_MIN = _f("MARGIN_MIN", "0.05")

# --- Timeouts (seconds) ---
CHAT_FIRST_TOKEN_TIMEOUT = _f("CHAT_FIRST_TOKEN_TIMEOUT", "30")
CHAT_TOTAL_TIMEOUT = _f("CHAT_TOTAL_TIMEOUT", "90")
WORKER_TIMEOUT = _f("WORKER_TIMEOUT", "120")
HA_TIMEOUT = _f("HA_TIMEOUT", "10")

# States cache TTL for easy_query entity lookup
STATES_TTL = _f("STATES_TTL", "30")
