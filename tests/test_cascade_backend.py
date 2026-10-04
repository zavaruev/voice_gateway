"""CascadeBackend contract: ack-timer, tag stripping, apology, escalation acks.

Mock pattern mirrors tests/test_tts_gate.py (fake aiohttp session/post/resp).
The router SSE stream is faked as an ordered list of event dicts.
"""

import asyncio
import json
from unittest import mock

import backends
from backends import CascadeBackend


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
    assert out == [CascadeBackend.ACK_COMPLEX, "Привет! Как дела?"]


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
