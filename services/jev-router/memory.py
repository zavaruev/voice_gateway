"""Qdrant memory: voice_turns (dialogue log) + voice_facts (extracted facts).

Written fire-and-forget by jev-router after every route; read by the L2
smolagents-worker via the qdrant_search tool. Vectors come from the same
Ollama qwen3-embedding:0.6b model (dim 1024, cosine) as the classifier, so
all collections share one embedding space.
"""

import logging
import time
import uuid

import aiohttp

import config

logger = logging.getLogger("router.memory")

DIM = 1024
COLLECTIONS = {
    "voice_turns": "User utterances + reply snippets + route metadata",
    "voice_facts": "Long-lived facts extracted from dialogue (preferences, names)",
}


async def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict | list | str]:
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.request(
            method, f"{config.QDRANT_URL}{path}", json=payload
        ) as resp:
            body: dict | list | str
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = await resp.text()
            return resp.status, body


async def ensure_collections() -> bool:
    """Idempotent collection creation (called at startup)."""
    ok = True
    for name, desc in COLLECTIONS.items():
        try:
            status, body = await _request(
                "PUT",
                f"/collections/{name}",
                {
                    "vectors": {"size": DIM, "distance": "Cosine"},
                    "description": desc,
                },
            )
            if status == 200:
                logger.info("qdrant collection %s ready", name)
            elif status == 409:
                logger.debug("qdrant collection %s already exists", name)
            else:
                logger.error("qdrant ensure %s -> %s: %s", name, status, body)
                ok = False
        except Exception as e:
            logger.error("qdrant ensure %s failed: %s", name, e)
            ok = False
    return ok


async def _embed(text: str) -> list[float] | None:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": [text]},
        ) as resp:
            if resp.status != 200:
                logger.error("memory embed http %s", resp.status)
                return None
            data = await resp.json()
            return data["embeddings"][0]


async def save_turn(
    text: str,
    reply: str,
    route: str,
    confidence: float,
    session_id: str,
    stream_name: str,
) -> None:
    """Fire-and-forget turn log. Never raises (caller wraps in create_task)."""
    try:
        vec = await _embed(text)
        if vec is None:
            return
        point = {
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {
                "type": "turn",
                "text": text[:500],
                "reply": reply[:500],
                "route": route,
                "confidence": confidence,
                "session_id": session_id,
                "stream_name": stream_name,
                "ts": time.time(),
            },
        }
        # Qdrant 1.19.1 in this env 404s on .../points/upsert (verified with
        # curl, both POST and PUT); plain PUT /points works and upserts by id.
        status, body = await _request(
            "PUT", "/collections/voice_turns/points?wait=true", {"points": [point]}
        )
        if status != 200:
            logger.error("memory save_turn -> %s: %s", status, body)
    except Exception as e:
        logger.error("memory save_turn failed: %s", e)


async def save_fact(fact: str, session_id: str = "") -> None:
    try:
        vec = await _embed(fact)
        if vec is None:
            return
        point = {
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {
                "type": "fact",
                "text": fact[:500],
                "session_id": session_id,
                "ts": time.time(),
            },
        }
        status, body = await _request(
            "PUT", "/collections/voice_facts/points?wait=true", {"points": [point]}
        )
        if status != 200:
            logger.error("memory save_fact -> %s: %s", status, body)
    except Exception as e:
        logger.error("memory save_fact failed: %s", e)


async def search(query: str, limit: int = 5) -> list[dict]:
    """Semantic search across both collections; returns merged payloads."""
    try:
        vec = await _embed(query)
        if vec is None:
            return []
        results: list[dict] = []
        for name in COLLECTIONS:
            status, body = await _request(
                "POST",
                f"/collections/{name}/points/search",
                {"vector": vec, "limit": limit, "with_payload": True},
            )
            if status != 200:
                logger.error("memory search %s -> %s: %s", name, status, body)
                continue
            for hit in body.get("result", []):
                pl = hit.get("payload") or {}
                pl["_score"] = hit.get("score", 0.0)
                pl["_collection"] = name
                results.append(pl)
        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return results[:limit]
    except Exception as e:
        logger.error("memory search failed: %s", e)
        return []
