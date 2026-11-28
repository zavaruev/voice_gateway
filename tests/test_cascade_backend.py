"""CascadeBackend contract: ack-timer, tag stripping, apology, escalation acks.

Mock pattern mirrors tests/test_tts_gate.py (fake aiohttp session/post/resp).
The router SSE stream is faked as an ordered list of event dicts.
"""

import asyncio
import json
from unittest import mock

import backends
from backends import CascadeBackend


def test_dobbi_persona_restyles_only_the_gateway_spoken_lines():
    """Complex-turn acknowledgement and the apology are gateway-written, so the
    house-elf's own turns belong there; router replies keep their own persona."""
    b = CascadeBackend("http://router:8091")
    assert b._ack_for("") == CascadeBackend.ACK_COMPLEX
    assert b._sorry_for("") == CascadeBackend.SORRY
    assert b._ack_for("dobbi", "сэр") == "Домовой слушает, сэр. Секунду."
    assert b._ack_for("dobbi", "мэм") == "Домовой слушает, мэм. Секунду."
    assert b._sorry_for("dobbi", "мэм") == "Домовой оплошал, мэм. Повторите, пожалуйста."
    assert b._sorry_for("dobbi", "") == "Домовой оплошал. Повторите, пожалуйста."


def _ev(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n"


class _FakeContent:
    def __init__(self, events, delay=0.0):
        self._lines = [_ev(e) for e in events] + [b"data: [DONE]\n"]
        self._delay = delay

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield line

        return gen()


class _FakeResp:
    status: int

    def __init__(self, events, status=200, delay=0.0):
        self.status = status
        self.content = _FakeContent(events, delay)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return "boom"


class _FakePost:
    def __init__(self, events, status, delay):
        self._events, self._status, self._delay = events, status, delay

    async def __aenter__(self):
        return _FakeResp(self._events, self._status, self._delay)

    async def __aexit__(self, *a):
        return False


class _FakeSess:
    def __init__(self, events, status=200, delay=0.0):
        self._events, self._status, self._delay = events, status, delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, *a, **k):
        return _FakePost(self._events, self._status, self._delay)


async def _run(events, status=200, delay=0.0, ack_delay=3.0):
    q = asyncio.Queue()
    be = CascadeBackend(router_url="http://x", ack_delay=ack_delay)
    fake = _FakeSess(events, status, delay)
    with mock.patch("backends.aiohttp.ClientSession", return_value=fake):
        await be.generate_response("тест", "sess", "test", q)
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _drain(items):
    assert items, "queue must not be empty"
    assert items[-1] is None, f"stream must end with None: {items!r}"
    return items[:-1]


def test_complex_route_immediate_ack():
    events = [
        {"type": "route", "route": "complex_logic", "confidence": 0.9},
        {"type": "progress", "text": "Анализирую запрос…"},
        {"type": "sentence", "text": "Проверил: всё работает."},
        {"type": "done", "route": "complex_logic", "elapsed": 5.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out[0] == CascadeBackend.ACK_COMPLEX
    assert "Анализирую" in out[1]
    assert out[2] == "Проверил: всё работает."


def test_suspect_route_speaks_the_router_cue_instead_of_the_generic_ack():
    """27.11.2026: the deterministic "I may have misheard" sentence is the
    acknowledgement. A second «Секунду, занимаюсь…» would bury it and waste
    three seconds before the user learns there was a problem."""
    cue = (
        "Похоже, я неверно расслышала. Если это не то — повторите, пожалуйста, "
        "правильную фразу."
    )
    events = [
        {"type": "route", "route": "expert", "confidence": 0.0,
         "reason": "stt_suspect"},
        {"type": "sentence", "text": "Полностью его звали Александр Сергеевич Пушкин."},
        {"type": "sentence", "text": cue},
        {"type": "done", "route": "expert", "elapsed": 5.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert CascadeBackend.ACK_COMPLEX not in out, out
    assert out[-1] == cue, out


def test_other_expert_routes_get_the_same_heard_ack():
    events = [
        {"type": "route", "route": "expert", "confidence": 0.0,
         "reason": "unresolved_question"},
        {"type": "sentence", "text": "Ответ Гермеса."},
        {"type": "done", "route": "expert", "elapsed": 5.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out[0] == CascadeBackend.ACK_HEARD, out


def test_expert_route_immediately_acks_that_the_question_was_heard():
    """27.11.2026: a Hermes-bound question must never sit in silence. The ack
    is explicit about the question having been heard, and it comes from the
    route event — not from a timer that may fire after the answer."""
    for reason in ("stt_suspect", "unresolved_question", "low_confidence"):
        events = [
            {"type": "route", "route": "expert", "confidence": 0.0,
             "reason": reason},
            {"type": "sentence", "text": "Ответ Гермеса."},
            {"type": "done", "route": "expert", "elapsed": 5.0},
        ]
        out = _drain(asyncio.run(_run(events)))
        assert out[0] == "Вопрос услышан. Секунду, думаю.", (reason, out)
        assert out[1] == "Ответ Гермеса.", (reason, out)
        assert CascadeBackend.ACK_COMPLEX not in out, (reason, out)


def test_dobbi_heard_ack_is_masculine_and_uses_the_honorific():
    b = CascadeBackend("http://x")
    assert b._ack_for_question("", "") == "Вопрос услышан. Секунду, думаю."
    assert b._ack_for_question("dobbi", "сэр") == (
        "Домовой услышал, сэр. Секунду думаю."
    )
    assert b._ack_for_question("dobbi", "") == "Домовой услышал. Секунду думаю."


def test_the_heard_ack_is_not_treated_as_an_answer_when_hermes_fails():
    """An ack must not disarm the apology: only a real answer sentence may."""
    events = [
        {"type": "route", "route": "expert", "confidence": 0.0,
         "reason": "stt_suspect"},
        {"type": "error", "message": "hermes down"},
        {"type": "done", "route": "expert", "elapsed": 5.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out[0] == "Вопрос услышан. Секунду, думаю.", out
    assert CascadeBackend.SORRY in out, out


def test_easy_route_fast_no_ack():
    events = [
        {"type": "route", "route": "easy_query", "confidence": 1.0},
        {"type": "sentence", "text": "Кухня: включено"},
        {"type": "done", "route": "easy_query", "elapsed": 0.4},
    ]
    out = _drain(asyncio.run(_run(events, ack_delay=3.0)))
    assert out == ["Кухня: включено"]


def test_slow_first_sentence_fires_thinking_ack():
    events = [
        {"type": "route", "route": "general_qa", "confidence": 0.9},
        {"type": "sentence", "text": "Отвечаю."},
        {"type": "done", "route": "general_qa", "elapsed": 1.0},
    ]
    out = _drain(asyncio.run(_run(events, delay=0.15, ack_delay=0.03)))
    assert out[0] == CascadeBackend.ACK_THINK
    assert out[1] == "Отвечаю."


def test_http_error_apologizes():
    out = _drain(asyncio.run(_run([], status=500)))
    assert out == [CascadeBackend.SORRY]


def test_error_after_ack_still_apologizes():
    # The ack already cancelled the watchdog; silence after it would be final.
    events = [
        {"type": "route", "route": "complex_logic", "confidence": 0.9},
        {"type": "error", "message": "worker down"},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.ACK_COMPLEX, CascadeBackend.SORRY]


def test_strips_emotion_tags():
    events = [
        {"type": "route", "route": "expert", "confidence": 0.9},
        {"type": "sentence", "text": "[happy] Привет! Как дела?"},
        {"type": "done", "route": "expert", "elapsed": 2.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.ACK_HEARD, "Привет! Как дела?"]


def test_escalation_second_route_event_acks_immediately():
    events = [
        {"type": "route", "route": "easy_query", "confidence": 0.6,
         "reason": "query_unresolved"},
        {"type": "route", "route": "complex_logic", "confidence": 0.6},
        {"type": "sentence", "text": "Не нашёл."},
        {"type": "done", "route": "complex_logic", "elapsed": 3.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out[0] == CascadeBackend.ACK_COMPLEX
    assert out[1] == "Не нашёл."


def test_empty_stream_apologizes():
    out = _drain(asyncio.run(_run([])))
    assert out == [CascadeBackend.SORRY]


# --- room hint (Telegram /room binding) ------------------------------------


def _run_captured(events, **kwargs):
    """Like _run, but returns (queue_items, the JSON payload POSTed)."""
    q = asyncio.Queue()
    be = CascadeBackend(router_url="http://x")
    fake = _FakeSess(events)
    captured = {}
    orig_post = fake.post

    def post(*a, **k):
        captured.update(k.get("json") or {})
        return orig_post(*a, **k)

    fake.post = post
    with mock.patch("backends.aiohttp.ClientSession", return_value=fake):
        asyncio.run(be.generate_response("тест", "sess", "test", q, **kwargs))
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out, captured


def test_room_hint_forwarded_to_router():
    events = [
        {"type": "route", "route": "easy_action", "confidence": 0.9},
        {"type": "sentence", "text": "Свет выключен."},
        {"type": "done", "route": "easy_action", "elapsed": 0.2},
    ]
    _out, payload = _run_captured(events, room="kitchen")
    assert payload["room"] == "kitchen"
    # The history key stays the unique stream_name — room must never
    # overwrite it (that would merge a chat's turns into a camera's ring).
    assert payload["stream_name"] == "test"


def test_room_omitted_when_empty():
    events = [{"type": "route", "route": "easy_action", "confidence": 0.9},
              {"type": "done", "route": "easy_action", "elapsed": 0.2}]
    _out, payload = _run_captured(events)
    # Devices/cameras send no room -> the payload is byte-identical to the
    # pre-Telegram contract.
    assert "room" not in payload


# --- progress heartbeats are NOT content (12.11.2026) -------------------------
#
# Measured on the 08:51 kitchen turn: the worker burned 120 s, the gateway
# spoke eight «Анализирую запрос…» heartbeats (one per 15 s), and the error
# event's apology was suppressed because the heartbeats had flipped the SAME
# `spoken` flag the apology checks. The user heard eight progress phrases
# and then nothing — and the camera watchdog stayed silent because the
# heartbeats kept the turn "alive". The ack already got this right («ack
# alone is not content»); progress gets the same treatment.

def test_progress_before_an_error_still_apologizes():
    events = [
        {"type": "route", "route": "complex_logic", "confidence": 0.9},
        {"type": "progress", "text": "Анализирую запрос…"},
        {"type": "error", "message": "timeout>120s"},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.ACK_COMPLEX, "Анализирую запрос…",
                   CascadeBackend.SORRY]


def test_a_progress_only_stream_that_ends_still_apologizes():
    # done with heartbeats and not one sentence: the turn produced nothing
    # speakable, so the promise the heartbeats made must be answered.
    events = [
        {"type": "route", "route": "complex_logic", "confidence": 0.9},
        {"type": "progress", "text": "Анализирую запрос…"},
        {"type": "done", "route": "complex_logic", "elapsed": 120.12},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.ACK_COMPLEX, "Анализирую запрос…",
                   CascadeBackend.SORRY]


def test_real_content_still_suppresses_the_apology():
    """Unchanged behaviour, guarded against over-correcting: once a real
    answer has been heard, a late error must NOT append an apology."""
    events = [
        {"type": "route", "route": "complex_logic", "confidence": 0.9},
        {"type": "progress", "text": "Анализирую запрос…"},
        {"type": "sentence", "text": "Готово: свет выключен."},
        {"type": "error", "message": "late failure"},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.ACK_COMPLEX, "Анализирую запрос…",
                   "Готово: свет выключен."]


def test_a_keepalive_event_is_silent_and_does_not_count_as_an_answer():
    """27.11.2026: the router keeps a long Hermes tool turn's SSE socket alive
    with a textless `keepalive` event. It must never be spoken, and it must
    not disarm the apology if Hermes then dies — only content does that."""
    events = [
        {"type": "route", "route": "expert", "confidence": 0.01,
         "reason": "download_request"},
        {"type": "keepalive"},
        {"type": "keepalive"},
        {"type": "error", "message": "hermes died"},
        {"type": "done", "route": "expert", "elapsed": 70.0},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == [CascadeBackend.SORRY], out


def test_a_download_request_is_not_acknowledged_before_its_fact():
    """A parseable download request has its own factual sentence. The generic
    «question heard, thinking» ack would add a second, misleading phrase."""
    events = [
        {"type": "route", "route": "expert", "confidence": 0.92,
         "reason": "download_request"},
        {"type": "sentence", "text": "Скачивается восьмой сезон Симпсонов."},
        {"type": "done", "route": "expert", "elapsed": 0.4},
    ]
    out = _drain(asyncio.run(_run(events)))
    assert out == ["Скачивается восьмой сезон Симпсонов."], out
