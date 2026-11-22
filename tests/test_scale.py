"""Kitchen scale (ESP32, 192.168.22.123) — deterministic L1 commands.

Pure tests: a stub `post` stands in for the MCP endpoint, so no device and no
network are needed. The router's service dir is put on sys.path exactly like
test_router_resolution.py does.
"""

import importlib.util
import json
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import resolver  # noqa: E402
import scale  # noqa: E402


class _StubPost:
    """Queue of JSON-RPC replies; records every (url, payload) pair."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, url, payload):
        self.calls.append((url, payload))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _rpc_result(text, is_error=False):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


def _weight(grams=0.0, zero=True, stable=True):
    return json.dumps({
        "grams": grams, "value": grams, "unit": "g", "decimals": 2,
        "stable": stable, "zero": zero, "tare": True, "sigma_mg": 100,
        "cal": 425.6, "tare_offset": 420205, "uptime": 100,
    })


# --- phrase recognition ----------------------------------------------------

def test_scale_phrases_require_weight_words():
    assert resolver.resolve_scale("сколько весит").kind == "weight"
    assert resolver.resolve_scale("Сколько весит?").kind == "weight"
    assert resolver.resolve_scale("какой вес на весах").kind == "weight"
    assert resolver.resolve_scale("что показывает весы").kind == "weight"
    assert resolver.resolve_scale("обнули весы").kind == "tare"
    assert resolver.resolve_scale("обнулить весы").kind == "tare"
    assert resolver.resolve_scale("тарируй весы").kind == "tare"
    assert resolver.resolve_scale("сделай тару").kind == "tare"
    # Other devices and ordinary commands must never be captured.
    assert resolver.resolve_scale("сколько весит пылесос") is None
    assert resolver.resolve_scale("включи свет") is None
    assert resolver.resolve_scale("сколько времени") is None


# --- spoken number + noun agreement ---------------------------------------

def test_weight_sentence_noun_agreement():
    cases = [
        (0, "ноль граммов"),
        (1, "один грамм"),
        (2, "два грамма"),
        (5, "пять граммов"),
        (11, "одиннадцать граммов"),
        (21, "двадцать один грамм"),
        (22, "двадцать два грамма"),
        (100, "сто граммов"),
        (123, "сто двадцать три грамма"),
    ]
    for grams, spelled in cases:
        sentence = scale.weight_sentence({"grams": grams, "zero": grams == 0})
        assert sentence is not None
        assert sentence.startswith("На весах ")
        assert spelled in sentence, (grams, sentence)


def test_weight_sentence_refuses_bad_payload():
    assert scale.weight_sentence(None) is None
    assert scale.weight_sentence({"grams": "не число"}) is None


# --- MCP client ------------------------------------------------------------

def test_get_weight_parses_rpc_content():
    stub = _StubPost(_rpc_result(_weight(grams=123.45, zero=False)))
    got = __import__("asyncio").run(scale.get_weight(post=stub))
    assert got["grams"] == 123.45
    assert stub.calls[0][0] == scale.SCALE_MCP_URL
    assert stub.calls[0][1]["method"] == "tools/call"
    assert stub.calls[0][1]["params"]["name"] == "get_weight"


def test_unreachable_scale_speaks_refusal():
    stub = _StubPost(OSError("connection refused"))
    assert __import__("asyncio").run(scale.handle("weight", post=stub)) == (
        "Весы не отвечают."
    )
    stub2 = _StubPost(OSError("connection refused"))
    assert __import__("asyncio").run(scale.handle("tare", post=stub2)) == (
        "Весы не отвечают."
    )


def test_tare_refuses_while_loaded():
    stub = _StubPost(_rpc_result(_weight(grams=250, zero=False)))
    ok, reason = __import__("asyncio").run(scale.tare_checked(post=stub))
    assert (ok, reason) == (False, "occupied")
    assert len(stub.calls) == 1  # the tare tool was NOT called
    sentence = scale.sentence_for("tare", reason=reason)
    assert "лежит" in sentence.lower()


def test_tare_verifies_zero_after_call():
    stub = _StubPost(
        _rpc_result(_weight(grams=0, zero=True)),
        _rpc_result("{}"),  # tare tool result
        _rpc_result(_weight(grams=0, zero=True)),
    )
    assert __import__("asyncio").run(scale.tare_checked(post=stub)) == (True, "ok")
    assert [c[1]["params"]["name"] for c in stub.calls] == [
        "get_weight", "tare", "get_weight",
    ]


def test_tare_without_verified_zero_is_not_a_success():
    stub = _StubPost(
        _rpc_result(_weight(grams=0, zero=True)),
        _rpc_result("{}"),
        _rpc_result(_weight(grams=0, zero=False)),
    )
    ok, reason = __import__("asyncio").run(scale.tare_checked(post=stub))
    assert (ok, reason) == (False, "unverified")
    assert "не получилось" in scale.sentence_for("tare", reason=reason).lower()


# --- routing integration ---------------------------------------------------

def _router_app():
    spec = importlib.util.spec_from_file_location(
        "jev_router_scale_app", os.path.join(_ROUTER, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scale_phrases_are_intercepted_before_classifier(monkeypatch):
    app = _router_app()

    def _must_not_run(*args, **kwargs):
        raise AssertionError("classifier must not run for a scale phrase")

    monkeypatch.setattr(app.classifier, "classify", _must_not_run)

    async def _handle_stub(kind, **kwargs):
        return "На весах сто граммов."

    monkeypatch.setattr(app.scale, "handle", _handle_stub)

    async def _drive():
        gen = app._handle(app.RouteRequest(
            text="сколько весит", session_id="probe", stream_name="kitchen"
        ))
        first = await gen.__anext__()
        second = await gen.__anext__()
        await gen.aclose()
        return json.loads(first.removeprefix("data: ")), json.loads(
            second.removeprefix("data: ")
        )

    route, sentence = __import__("asyncio").run(_drive())
    assert route["type"] == "route"
    assert route["route"] == "scale"
    assert route["reason"] == "scale_weight"
    assert route["confidence"] == 1.0
    assert sentence["type"] == "sentence"
    assert sentence["text"] == "На весах сто граммов."
