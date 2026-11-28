"""A download request is a real action, not misheard speech; and a long
Hermes tool turn must keep the router->gateway SSE socket alive.

Field case 27.11.2026 13:10 MSK, living room: «скачай восьмой сезон
Симпсонов» was classified `stt_suspect`, the gateway's `sock_read=60 s`
fired while Hermes was still searching Rutracker, and the room apologized —
while the torrent was actually added at 13:13:56 MSK and is downloading.
Two independent faults, two independent behaviours pinned here.
"""
import asyncio
import json
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

from resolver import is_unrecognised_utterance  # noqa: E402


def _decision(route, conf, reason):
    return type("D", (), {"route": route, "confidence": conf, "reason": reason})()


async def _drive(router, text, stream="livingroom"):
    return [
        json.loads(c.removeprefix("data: "))
        async for c in router._handle(router.RouteRequest(
            text=text, session_id="probe", stream_name=stream
        ))
    ]


def test_a_download_command_is_not_misheard_speech():
    """«скачай ... сезон» is a verb plus an object, not a garbled question."""
    assert is_unrecognised_utterance("скачай восьмой сезон Симпсонов") is False


def test_a_real_mangled_question_is_still_suspect():
    """The 27.11 10:15 field case keeps its old path — the rule is narrow."""
    assert is_unrecognised_utterance("так звали Пушкина") is True


@pytest.mark.asyncio
async def test_a_download_request_never_gets_a_misheard_cue_or_a_tool_agent(monkeypatch):
    """A definite refusal is one short sentence. Hermes used to guess a show
    from earlier turns and narrate every step; that path is gone."""
    import app as router

    chat_kwargs, worker_calls = [], []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _worker(*a, **k):
        worker_calls.append(1)
        yield {"type": "sentence", "text": "Уточните, что скачать."}

    async def _chat(*a, **k):
        chat_kwargs.append(k)
        yield "Возможно, вы имели в виду Симпсонов?", ""

    async def _search(req, *, url=None, post=None):
        return None  # definite refusal — no retry, no guessing

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router, "_run_worker", _worker)
    monkeypatch.setattr(router.download, "search_and_download", _search)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "скачай восьмой сезон Симпсонов")
    routes = [e for e in events if e.get("type") == "route"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert not worker_calls, "the download was handed to a tool agent with no download tool"
    assert not chat_kwargs, "Hermes must not guess a title for a download request"
    assert routes and routes[0]["route"] == "expert", routes
    assert routes[0]["reason"] == "download_request", routes
    assert sentences == ["Пытаюсь скачать.", "Не нашла подходящего релиза."], sentences
    assert router._STT_SUSPECT_ACK not in sentences, sentences


@pytest.mark.asyncio
async def test_a_slow_expert_call_emits_silent_keepalives(monkeypatch):
    """While Hermes runs tools the gateway sees bytes on the SSE socket.
    Keepalives carry NO text: only real content counts as an answer, and the
    gateway must not speak 'still working' every fifteen seconds."""
    import app as router

    monkeypatch.setattr(router, "EXPERT_KEEPALIVE_S", 0.02)

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.0, "ambiguous(0.00<0.55)")

    async def _chat(*a, **k):
        await asyncio.sleep(0.1)
        yield "Ответ Гермеса.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "как звали Пушкина?")
    keepalives = [e for e in events if e.get("type") == "keepalive"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert len(keepalives) >= 2, events
    assert all(set(e) == {"type"} for e in keepalives), keepalives
    assert sentences == ["Ответ Гермеса."], sentences


@pytest.mark.asyncio
async def test_a_fast_expert_call_has_no_keepalive(monkeypatch):
    """A normal 0.1 s Hermes answer must not gain extra SSE events."""
    import app as router

    monkeypatch.setattr(router, "EXPERT_KEEPALIVE_S", 5.0)

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.0, "ambiguous(0.00<0.55)")

    async def _chat(*a, **k):
        yield "Ответ Гермеса.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "как звали Пушкина?")
    assert not [e for e in events if e.get("type") == "keepalive"], events
