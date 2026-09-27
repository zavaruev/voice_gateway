"""Central env configuration for smolagents-worker (cascade level 2).

PURPOSE
    One import-time configuration surface for this service: every URL,
    credential and timeout is read from the environment here, once, so
    app.py / tools.py / honesty.py just `import config` and no address or
    secret is ever written into the code (the repository is PUBLIC — env
    vars only, defaults below point at the home LAN and are overridden in
    docker-compose.yml `environment:`).

ROLE IN THE CASCADE
    ESP32 satellite -> main.py (FastAPI :6050, audio/VAD/STT/TTS)
    -> jev-router (L1, :8091) -> THIS SERVICE (L2, :8092, POST /invoke)
    -> Hermes (L3 expert) / Home Assistant (MCP+REST) / Qdrant / Ollama.
    Each URL constant below is exactly one hop of that chain.

ENV VARS (all optional; override before the process starts)
    OMNIROUTE_URL, OMNIROUTE_COMBO   primary FREE LLM, OpenAI-compatible /v1
    HERMES_URL, HERMES_API_KEY, HERMES_MODEL   L3 failover + hermes_expert
    HA_URL, HA_TOKEN                 Home Assistant REST + /api/mcp (Bearer)
    ROUTER_URL                       jev-router (/weather, SSE /route)
    QDRANT_URL, OLLAMA_URL, EMBED_MODEL   dialogue memory (embed + search)
    WORKER_TIMEOUT       hard cap per /invoke request, seconds
    MAX_STEPS            CodeAgent step ceiling — bounds a looping model
    HEARTBEAT_INTERVAL   SSE progress period, kept <= 20 s so the turn
                         watchdog upstream hears a live gateway instead of
                         silence (replaces the old 30 s watchdog)
    LLM_CALL_TIMEOUT     per-completion OpenAI timeout (plus 1 retry)
    TOOL_TIMEOUT         per-tool HTTP budget (ha_*)
    EXPERT_TIMEOUT       hermes_expert budget — L3 diagnostics are slow

CONTRACT / FAILURE MODES
    Plain module-level constants evaluated at import. `_f`/`_i` raise
    ValueError on a malformed numeric env value — deliberately fail fast at
    startup instead of misbehaving mid-conversation. Ports 8091/8092 carry
    no auth by deliberate decision (home LAN only): never expose them.
"""

import os


def _f(name: str, default: str) -> float:
    """Env var as float (falls back to `default` string). Raises ValueError
    at import time if the variable is set but not numeric."""
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    """Env var as int. Goes through float first so values like "15.0"
    (a compose file may render numbers that way) still parse."""
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
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8091").rstrip("/")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.22.102:6333").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.22.102:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# --- Runtime limits ---
WORKER_TIMEOUT = _f("WORKER_TIMEOUT", "120")     # hard cap per /invoke request
MAX_STEPS = _i("MAX_STEPS", "15")                # CodeAgent steps ceiling
HEARTBEAT_INTERVAL = _f("HEARTBEAT_INTERVAL", "15")  # progress SSE, <=20 s
LLM_CALL_TIMEOUT = _f("LLM_CALL_TIMEOUT", "60")  # per-completion OpenAI timeout
TOOL_TIMEOUT = _f("TOOL_TIMEOUT", "20")          # per-tool HTTP (ha_*)
# Must stay well below the gateway's WATCHDOG_TIMEOUT (90 s): a tool that
# blocks for longer than that guarantees the satellite has already spoken
# the "I'm stuck, please repeat" apology before the real answer exists.
# 2026-09-27: hermes_expert burned 92 s of a 90 s budget on a one-word
# utterance, pushing the whole turn to 104 s and turning a clarifying
# question into a dead turn. 25 s keeps the expert useful without letting
# it eat the turn.
EXPERT_TIMEOUT = _f("EXPERT_TIMEOUT", "25")      # hermes_expert tool
