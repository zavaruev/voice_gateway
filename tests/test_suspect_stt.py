"""Suspect-STT handling: a fluent but wrong transcript must be recognised as
"no intent at all" before any word-specific repair is attempted.

Field case 27.11.2026 10:15 MSK: Whisper wrote «Компьютер так звали Пушкина»
from «как звали Пушкина?». There was loud real speech in the window, so every
acoustic gate passed; the classifier said complex_logic/ambiguous(0.00),
`is_question_shape()` said False, and the tool agent burned 120 s in silence.
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


def test_a_mangled_question_with_no_intent_is_suspect():
    assert is_unrecognised_utterance("так звали Пушкина") is True


def test_the_same_question_with_its_marker_is_not_suspect():
    assert is_unrecognised_utterance("как звали Пушкина?") is False


def test_a_light_command_is_not_suspect():
    assert is_unrecognised_utterance("выключи свет") is False


def test_a_bare_media_verb_is_not_suspect():
    assert is_unrecognised_utterance("пауза") is False


def test_a_device_question_is_not_suspect():
    assert is_unrecognised_utterance("какой фильм мы смотрели?") is False


def test_a_device_without_a_verb_is_not_suspect():
    """«и свет» must keep its short clarification, never go to Hermes."""
    assert is_unrecognised_utterance("и свет") is False


@pytest.mark.asyncio
async def test_a_mangled_utterance_goes_straight_to_hermes_with_the_cue(monkeypatch):
    """The exact field transcript must never reach L2 and must end with the
    spoken "I may have misheard" cue (the last sentence opens the window)."""
    import app as router

    worker_calls, chat_kwargs = [], []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.0, "ambiguous(0.00<0.55)")

    async def _worker(*a, **k):
        worker_calls.append(1)
        yield {"type": "sentence", "text": "Ответ инструмента."}

    async def _chat(*a, **k):
        chat_kwargs.append(k)
        yield "Полностью его звали Александр Сергеевич Пушкин.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router, "_run_worker", _worker)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "так звали Пушкина")
    routes = [e for e in events if e.get("type") == "route"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert not worker_calls, "the suspect turn reached the tool agent"
    assert routes and routes[0]["route"] == "expert", routes
    assert routes[0]["reason"] == "stt_suspect", routes
    assert chat_kwargs and chat_kwargs[0].get("stt_suspect") is True, chat_kwargs
    assert sentences[0].startswith("Полностью его звали"), sentences
    assert sentences[-1] == router._STT_SUSPECT_ACK, sentences


@pytest.mark.asyncio
async def test_a_question_with_its_marker_keeps_the_normal_hermes_route(monkeypatch):
    """The v2.87 unresolved-question path is unchanged: no suspect cue."""
    import app as router

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.0, "ambiguous(0.00<0.55)")

    async def _chat(*a, **k):
        yield "Ответ Гермеса.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "как звали Пушкина?")
    routes = [e for e in events if e.get("type") == "route"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert routes[0]["route"] == "expert", routes
    assert routes[0]["reason"] != "stt_suspect", routes
    assert router._STT_SUSPECT_ACK not in sentences, sentences


@pytest.mark.asyncio
async def test_a_device_command_with_no_confidence_stays_on_the_deterministic_ladder(monkeypatch):
    """A lamp phrase with conf=0 is not 'misheard speech' — it is a command
    L2 must resolve, and the old ladder must keep running."""
    import app as router

    worker_calls = []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.0, "low_confidence(0.00<0.85)")

    async def _worker(*a, **k):
        worker_calls.append(1)
        yield {"type": "sentence", "text": "Уточните комнату."}

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router, "_run_worker", _worker)

    events = await _drive(router, "выключи свет")
    routes = [e for e in events if e.get("type") == "route"]

    assert worker_calls, "the lamp command was stolen from L2"
    assert routes[0]["route"] == "complex_logic", routes
    assert routes[0]["reason"] != "stt_suspect", routes


def test_the_cue_opens_the_followup_window_without_a_peak_gate():
    """The correction must land in the camera's existing follow-up window:
    «повторите, пожалуйста» is already an interrogative to the player, and a
    question window accepts a normally-spoken answer without a peak gate."""
    import app as router
    import camera_client

    assert camera_client._is_question_reply(router._STT_SUSPECT_ACK) is True


def test_dobbi_gets_the_masculine_suspect_cue():
    import app as router

    assert router.dobbi_style(router._STT_SUSPECT_ACK, "сэр") == (
        "Домовой неверно расслышал, сэр. Повторите, пожалуйста, правильную фразу."
    )
    assert router.dobbi_style(router._STT_SUSPECT_ACK, "") == (
        "Домовой неверно расслышал. Повторите, пожалуйста, правильную фразу."
    )


@pytest.mark.asyncio
async def test_the_hermes_prompt_is_told_the_transcript_may_be_wrong(monkeypatch):
    import chat_proxy

    captured = {}

    async def _sentences(url, model, messages, api_key=""):
        captured["system"] = messages[0]["content"]
        yield "Ответ.", ""

    monkeypatch.setattr(chat_proxy, "stream_sentences", _sentences)
    out = [s async for s, _ in chat_proxy.stream_chat(
        "так звали Пушкина", expert=True, knowledge=True, stt_suspect=True
    )]

    assert out == ["Ответ."]
    assert "распознан" in captured["system"], captured
    assert "уточни" in captured["system"], captured
