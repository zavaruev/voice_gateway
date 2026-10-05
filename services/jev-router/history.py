"""Short-term dialogue memory: the last few exchanges of ONE satellite.

WHY THIS EXISTS (field case 28.09.2026): L2 sees one utterance at a time, so
an anaphora had nothing to resolve — «Выключи её» (said 4 s after the coffee
maker was reported on) made the free model search long-term Qdrant memory for
«устройства женского рода» and call HassTurnOff(area=спальня, domain=light):
it nearly switched off a bedroom lamp instead of the device the previous turn
was about. That call failed only by luck (the area word was RU while HA wants
the registry display name).

Long-term memory is not a substitute: it is vector-searched by a query the
model has to invent, it is written AFTER the turn, and it does not survive
«said ten seconds ago» reliably. This module is a plain in-process ring of
finished turns, read by app._handle BEFORE the turn mutates it, and handed to
the worker as the «Предыдущие реплики» block.

CONTRACT
    push(key, text, reply) — record one finished turn; key is the satellite
        (stream_name, falling back to session_id), "" is ignored.
    block(key) -> str      — one RU line per recent turn, oldest first, for
        the worker's task; "" when nothing recent. Reads do NOT mutate.
    reset()                — tests only.
    TTL: turns older than TTL_S are dropped (a «выключи её» answered from an
        hour-old exchange is worse than no context at all); MAX_TURNS is the
        hard cap per satellite, _MAX_STREAMS the cold-eviction bound.

Single event loop, no lock (app._handle is a generator on that loop); NOT
persisted on purpose — starting a dialogue from scratch after a restart is
harmless, replaying a stale one is not.
"""

import time
from collections import deque

# How many finished turns are kept per satellite. Four covers a pronoun that
# refers back two questions («а та, что в кухне?») without ever letting the
# agent act on a turn older than the current session's attention span.
MAX_TURNS = 4

# Turns older than this are not context any more, they are stale state.
TTL_S = 600.0

# Cold-eviction bound for the whole process (a satellite that stops talking
# must not pin its history forever; ~30 rooms is far beyond a home LAN).
_MAX_STREAMS = 32

# key -> deque of (monotonic-ish wall ts, user text, spoken reply)
_store: dict[str, deque] = {}


def _prune(dq: deque, now: float) -> None:
    """Drop expired (oldest-first) entries from one ring."""
    while dq and now - dq[0][0] > TTL_S:
        dq.popleft()


def push(key: str, text: str, reply: str) -> None:
    """Record one finished turn. Called by app._handle after every answer."""
    if not key:
        return
    dq = _store.get(key)
    if dq is None:
        if len(_store) >= _MAX_STREAMS:
            # Evict the least recently updated satellite first.
            cold = min(
                _store, key=lambda k: _store[k][-1][0] if _store[k] else 0.0
            )
            _store.pop(cold, None)
        dq = deque(maxlen=MAX_TURNS)
        _store[key] = dq
    dq.append((time.time(), (text or "").strip(), (reply or "").strip()))


def last_text(key: str) -> str:
    """The most recent finished turn's USER text, or "".

    Separate from `block()` on purpose: the block is prose for the LLM, and
    feeding it to a deterministic resolver would be both lossy and dangerous — it
    contains the previous REPLY too, so a resolver scanning it for verbs would
    act on what the room said rather than on what the user asked. Only the
    user's own words belong in a deterministic decision.
    """
    if not key:
        return ""
    dq = _store.get(key)
    if not dq:
        return ""
    now = time.time()
    _prune(dq, now)
    for _ts, t, _r in reversed(dq):
        if t:
            return t
    return ""


def block(key: str) -> str:
    """Recent turns as `- пользователь: «…» — ответ: «…»` lines, or "".

    Text-only, no label: the worker owns the wording around it (it knows how
    the task template reads) while this module owns the data.
    """
    if not key:
        return ""
    dq = _store.get(key)
    if not dq:
        return ""
    now = time.time()
    _prune(dq, now)
    if not dq:
        _store.pop(key, None)
        return ""
    return "\n".join(
        f"- пользователь: «{t}» — ответ: «{r}»"
        for _ts, t, r in dq
        if t
    )


def reset() -> None:
    """Drop everything (tests only)."""
    _store.clear()
