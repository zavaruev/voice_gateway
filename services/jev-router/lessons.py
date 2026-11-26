"""Self-learning rewrites: upper-level success becomes an L1 phrase rule.

A lesson is NEVER free text that the model could turn into an action. It is one
narrow rewrite of the USER's own words into a phrase L1 already understands:
«включи кафеварку» -> «включи кофеварку». The candidate comes from L2, is
validated offline against turns that already succeeded, and is applied only
after the upper level actually succeeded (see should_commit). Qdrant holds the
rules; a failed rule is removed on the next escalation, so the loop ends.
"""

import logging
import os
import re
import time

import memory

logger = logging.getLogger("lessons")

ENABLED = os.getenv("LEARN_ENABLED", "1").strip().lower() not in ("0", "false", "no")
COLLECTION = "voice_lessons"
MAX_ACTIVE = 200
MAX_PHRASE = 120
MAX_FROM = 120
# No code, no URLs, no tool names in a learned phrase. The vacuum incident
# showed what happens when remembered text is allowed to become a command.
_FORBIDDEN = re.compile(
    r"[<>{}\[\]\\;=`]|(?:\b)(?:import|eval|exec|__import__|ha_action|http|https)\b",
    re.IGNORECASE,
)
_ACTIVE: list[dict] = []
# (stream, normalized text) -> (timestamp, lesson_id of that successful turn)
_RECENT: dict[tuple[str, str], tuple[float, str]] = {}
_REPEAT_WINDOW_S = 600.0


def is_repeat(stream: str, text: str) -> str | None:
    """Lesson id of a recent identical turn, if the user had to ask twice."""
    key = (stream or "", (text or "").strip().lower())
    hit = _RECENT.get(key)
    if hit and time.time() - hit[0] < _REPEAT_WINDOW_S:
        return hit[1] or ""
    return None


def note_turn(stream: str, text: str, lesson_id: str = "") -> None:
    key = (stream or "", (text or "").strip().lower())
    _RECENT[key] = (time.time(), lesson_id)
    if len(_RECENT) > 512:
        now = time.time()
        for k, v in list(_RECENT.items()):
            if now - v[0] > _REPEAT_WINDOW_S:
                _RECENT.pop(k, None)


def rewrite(text: str) -> tuple[str, str | None]:
    """Return (text with a learned rewrite applied, lesson_id or None)."""
    t = (text or "").strip()
    if not ENABLED or not t:
        return t, None
    low = t.lower()
    for rule in _ACTIVE:
        if low == str(rule.get("from_text", "")).strip().lower():
            return str(rule.get("to_text", "")), str(rule.get("lesson_id", ""))
    return t, None


def validate(candidate: dict, history: list[dict]) -> tuple[bool, str]:
    """Pure, offline checks. history rows: {text, route} for successful turns."""
    if not candidate:
        return False, "empty"
    src = str(candidate.get("from_text") or "").strip()
    dst = str(candidate.get("to_text") or "").strip()
    if not (3 <= len(src) <= MAX_FROM) or not (3 <= len(dst) <= MAX_PHRASE):
        return False, "phrase length"
    if src.lower() == dst.lower():
        return False, "no change"
    if _FORBIDDEN.search(src) or _FORBIDDEN.search(dst):
        return False, "forbidden token"
    if candidate.get("source_level") not in ("l2", "l3"):
        return False, "source"
    if not history:
        return True, "ok"
    # A rewrite that would change a turn which already had a correct answer is
    # dangerous even when it looks like a harmless synonym.
    for row in history:
        if str(row.get("text") or "").strip().lower() == src.lower():
            return False, "history: would change an already successful turn"
    return True, "ok"


def should_commit(signals: dict) -> tuple[bool, str]:
    """All success conditions must hold; a single miss keeps the rule pending."""
    if signals.get("error"):
        return False, "error event"
    if signals.get("veto"):
        return False, "honesty veto"
    if not signals.get("sentence"):
        return False, "nothing was spoken"
    if not signals.get("done"):
        return False, "no clean done event"
    if signals.get("repeated"):
        return False, "the same request repeated"
    outcome = signals.get("outcome") or {}
    if not outcome.get("confirmed"):
        return False, "upper-level outcome not confirmed"
    return True, "ok"


def mark_failed(lesson_id: str, reason: str = "") -> dict | None:
    """Quarantine a rule locally; persistence is fire-and-forget in fail()."""
    global _ACTIVE
    hit = next((r for r in _ACTIVE if r.get("lesson_id") == lesson_id), None)
    _ACTIVE = [r for r in _ACTIVE if r.get("lesson_id") != lesson_id]
    if hit:
        logger.warning("lesson %s failed: %s", lesson_id, reason)
    return hit


async def _points() -> list[dict]:
    try:
        status, body = await memory._request(
            "POST", f"/collections/{COLLECTION}/points/scroll",
            {"limit": MAX_ACTIVE * 2, "with_payload": True},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson scroll failed: %r", e)
        return []
    if status != 200 or not isinstance(body, dict):
        return []
    result = body.get("result") or {}
    return [p for p in (result.get("points") or []) if isinstance(p, dict)]


async def refresh() -> int:
    """Load active rules into the in-memory cache. Never raises."""
    global _ACTIVE
    rules = []
    for p in await _points():
        payload = p.get("payload") or {}
        if payload.get("status") == "active":
            rules.append(payload)
    _ACTIVE = rules[:MAX_ACTIVE]
    return len(_ACTIVE)


async def ensure_collection() -> None:
    try:
        status, _ = await memory._request(
            "PUT", f"/collections/{COLLECTION}",
            {"vectors": {"size": memory.DIM, "distance": "Cosine"}},
        )
        if status in (200, 409):
            logger.info("lesson collection ready")
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson collection ensure failed: %r", e)


async def save(payload: dict) -> None:
    """Persist a new candidate (status=pending). Never raises."""
    if not ENABLED:
        return
    try:
        vec = await memory._embed(payload["from_text"])
        if vec is None:
            return
        point = {
            "id": payload["lesson_id"], "vector": vec,
            "payload": {**payload, "status": "pending", "ts": time.time()},
        }
        status, body = await memory._request(
            "PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": [point]}
        )
        if status != 200:
            logger.warning("lesson save -> %s: %s", status, str(body)[:200])
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson save failed: %r", e)


async def set_status(lesson_id: str, status: str, reason: str = "") -> None:
    try:
        await memory._request(
            "POST", f"/collections/{COLLECTION}/points/payload?wait=true",
            {"points": [lesson_id],
             "payload": {"status": status, "reason": reason[:200], "ts": time.time()}},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson status update failed: %r", e)
    if status == "active":
        payload = next((r for r in await _points()
                        if r.get("payload", {}).get("lesson_id") == lesson_id), None)
        if payload and len(_ACTIVE) < MAX_ACTIVE:
            _ACTIVE.append(payload["payload"])
    else:
        mark_failed(lesson_id, reason)


_HISTORY_DAYS = 7.0
_HISTORY_TOTAL = 60
_HISTORY_PER_ROOM = 8
_DETERMINISTIC_ROUTES = {"easy_action", "easy_query", "calc", "scale"}


def select_history(
    points: list[dict], *, now: float | None = None,
    per_room: int = _HISTORY_PER_ROOM, total: int = _HISTORY_TOTAL,
) -> list[dict]:
    """Recent L1-answered turns — the corpus a new rewrite is tested against.

    A turn with `route=complex_logic` was an ESCALATION, not proof that the
    answer was right; 80 of the last 200 stored turns were escalations spanning
    59 days. Such a corpus would reject useful fixes for arbitrary old phrases.
    """
    now = time.time() if now is None else now
    per_stream: dict[str, int] = {}
    rows: list[dict] = []
    for point in points:
        pl = point.get("payload") or {}
        ts = float(pl.get("ts") or 0)
        route = str(pl.get("route") or "")
        if not (pl.get("text") and pl.get("reply")):
            continue
        if route not in _DETERMINISTIC_ROUTES:
            continue
        if ts <= 0 or now - ts > _HISTORY_DAYS * 86400:
            continue
        stream = str(pl.get("stream_name") or "")
        if per_stream.get(stream, 0) >= per_room:
            continue
        per_stream[stream] = per_stream.get(stream, 0) + 1
        rows.append({"text": pl["text"], "route": route})
        if len(rows) >= total:
            break
    return rows


async def successful_history() -> list[dict]:
    """A week of deterministic L1 turns, not an arbitrary tail of the log."""
    try:
        status, body = await memory._request(
            "POST", "/collections/voice_turns/points/scroll",
            {"limit": 500, "with_payload": True},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson history failed: %r", e)
        return []
    if status != 200 or not isinstance(body, dict):
        return []
    points = (body.get("result") or {}).get("points") or []
    return select_history(points)
