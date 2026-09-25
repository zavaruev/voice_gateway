"""Central env configuration for smolagents-worker (cascade level 2)."""

import os


def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)))


# --- LLM for code generation (cloud OmniRoute combo, failover -> Hermes) ---
OMNIROUTE_URL = os.getenv("OMNIROUTE_URL", "http://192.168.22.101:20128").rstrip("/")
OMNIROUTE_COMBO = os.getenv("OMNIROUTE_COMBO", "gemma4_31b_free")

HERMES_URL = os.getenv("HERMES_URL", "http://192.168.22.102:8642").rstrip("/")
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")
HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes-agent")

# --- Tool backends ---
HA_URL = os.getenv("HA_URL", "http://192.168.22.111:8123").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.22.102:6333").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.22.102:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# --- Runtime limits ---
WORKER_TIMEOUT = _f("WORKER_TIMEOUT", "120")     # hard cap per /invoke request
MAX_STEPS = _i("MAX_STEPS", "15")                # CodeAgent steps ceiling
HEARTBEAT_INTERVAL = _f("HEARTBEAT_INTERVAL", "15")  # progress SSE, <=20 s
LLM_CALL_TIMEOUT = _f("LLM_CALL_TIMEOUT", "60")  # per-completion OpenAI timeout
TOOL_TIMEOUT = _f("TOOL_TIMEOUT", "20")          # per-tool HTTP (ha_*)
EXPERT_TIMEOUT = _f("EXPERT_TIMEOUT", "90")      # hermes_expert tool
