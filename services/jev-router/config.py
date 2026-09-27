"""Central env configuration for jev-router (cascade level 1).

PURPOSE
  Single place where every tunable of the L1 router is read from the
  environment. jev-router sits between the voice gateway (`main.py`, port
  6050) and the escalation targets: smolagents-worker (L2, port 8092),
  Hermes (L3) and Home Assistant (MCP/REST). Everything the router talks to
  is a LAN service, so all URLs below are *overridable env vars* — the
  literal values are only host-network fallbacks for a bare `python app.py`
  run; the compose file passes the real addresses (and the HA/Hermes
  credentials) per deployment. Secrets (HA_TOKEN, HERMES_API_KEY) must never
  be pasted into this file: the repository is public.

  Contract: this module is imported at module import time by app.py,
  classifier.py, ha_client.py, chat_proxy.py, weather.py and memory.py, so
  a bad value here fails the whole service at startup (by design — better
  than silently routing to the wrong host).

Key groups (see sections below):
  * external service bases (Ollama / Qdrant / HA / OmniRoute / Hermes / L2);
  * classification thresholds calibrated on qwen3-embedding:0.6b;
  * timeouts in seconds for chat, worker and HA calls;
  * STATES_TTL — the TTL of both HA caches in ha_client (states + areas).
"""

import os


def _f(name: str, default: str) -> float:
    """Read a float env var.

    Exists so thresholds/timeouts stay numeric no matter how they are
    passed (`CONFIDENCE_THRESHOLD=0.85` in YAML is a string). Raises
    ValueError at import time on a malformed value — a hard startup failure
    is preferred over a silently mis-calibrated classifier.
    """
    return float(os.getenv(name, default))


# --- External services (all host-LAN addresses; service runs host-network) ---
# Hosts come from env in production (see docker-compose); the literals below
# are fallbacks for a manual run on this LAN. Tokens default to empty strings
# on purpose — real credentials are injected by the environment, never code.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.22.102:11434").rstrip("/")
# 1024-dim embedding model, shared with memory.py so the classifier and the
# dialogue memory search the same vector space.
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.22.102:6333").rstrip("/")

HA_URL = os.getenv("HA_URL", "http://192.168.22.111:8123").rstrip("/")
# Bearer token for both HA REST and the MCP endpoint (HA 2026.9.3).
HA_TOKEN = os.getenv("HA_TOKEN", "")

OMNIROUTE_URL = os.getenv("OMNIROUTE_URL", "http://192.168.22.101:20128").rstrip("/")
# Free OmniRoute combo model used as the general_qa brain and as the expert
# failover when Hermes is down.
OMNIROUTE_COMBO = os.getenv("OMNIROUTE_COMBO", "gemma4_31b_free")

HERMES_URL = os.getenv("HERMES_URL", "http://192.168.22.102:8642").rstrip("/")
# L3 expert + weather fallback; empty key means "trust the LAN endpoint".
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")

# L2 = smolagents-worker. localhost is correct: compose runs host-network,
# so the router reaches 8092 without a container DNS name.
WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8092").rstrip("/")

# --- Routing thresholds (calibrated on qwen3-embedding:0.6b, Sep 2026) ---
# Positive same-intent paraphrases: 0.60..0.98 cosine; cross-route pairs: 0.34..0.61.
# Confidence = clip((score - 0.50) / 0.40, 0, 1) => conf 0.85 requires score >= 0.84.
# Below threshold/margin the classifier escalates to complex_logic instead of
# guessing: a wrong easy_action would be a real side-effect in the house.
CONFIDENCE_THRESHOLD = _f("CONFIDENCE_THRESHOLD", "0.85")
MARGIN_MIN = _f("MARGIN_MIN", "0.05")

# --- Timeouts (seconds) ---
# SSE sock_read for chat streams: the first token must arrive within this,
# otherwise the failover target gets its turn (voice cannot wait forever).
CHAT_FIRST_TOKEN_TIMEOUT = _f("CHAT_FIRST_TOKEN_TIMEOUT", "30")
# Whole expert/general_qa budget including failover: Hermes-first, then
# OmniRoute (CHAT_TOTAL_TIMEOUT is the cascade's 90 s promise to the caller).
CHAT_TOTAL_TIMEOUT = _f("CHAT_TOTAL_TIMEOUT", "90")
# L2 CodeAgent cap; matches the worker's own 120 s hard limit plus its
# progress heartbeats, so the router never aborts before the worker does.
WORKER_TIMEOUT = _f("WORKER_TIMEOUT", "120")
# Per MCP/REST call: a slow HA must escalate to L2, not hang the voice turn.
HA_TIMEOUT = _f("HA_TIMEOUT", "10")

# States cache TTL for easy_query entity lookup. Also reused by
# HAClient.get_entity_areas() — the area map changes as rarely as states.
STATES_TTL = _f("STATES_TTL", "30")
