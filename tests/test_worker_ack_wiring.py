"""Wiring test for L2's verified-action acknowledgement.

Runs only where smolagents is installed (the worker image). On the gateway test
image it skips, because importing the worker's app.py there needs a dependency
that image deliberately does not ship.
"""

import json
import os
import sys

import pytest

pytest.importorskip("smolagents")

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
if _WORKER not in sys.path:
    sys.path.insert(0, _WORKER)

import app as worker_app  # noqa: E402
import tools  # noqa: E402


def _events(chunks):
    out = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


@pytest.mark.asyncio
async def test_verified_ack_is_yielded_before_the_final_answer(monkeypatch):
    def _fake_run(text, context, hist):
        box = tools._ACK_BOX.get()
        assert box is not None, "the SSE loop must set a per-request ack box"
        box.append("Включаю.")
        return "Кофеварка включена."

    monkeypatch.setattr(worker_app, "_run_agent", _fake_run)
    req = worker_app.InvokeRequest(
        text="включи кофеварку", session_id="t", stream_name="kitchen"
    )
    events = _events([c async for c in worker_app._handle(req)])
    sentences = [e["text"] for e in events if e.get("type") == "sentence"]
    assert sentences[0] == "Включаю.", (
        f"the verified ack must come first, got {sentences}"
    )
    assert "Кофеварка включена." in sentences


@pytest.mark.asyncio
async def test_no_ack_when_the_action_was_not_verified(monkeypatch):
    def _fake_run(text, context, hist):
        return "Не получилось включить."

    monkeypatch.setattr(worker_app, "_run_agent", _fake_run)
    req = worker_app.InvokeRequest(
        text="включи кофеварку", session_id="t", stream_name="kitchen"
    )
    events = _events([c async for c in worker_app._handle(req)])
    sentences = [e["text"] for e in events if e.get("type") == "sentence"]
    assert sentences == ["Не получилось включить."]


@pytest.mark.asyncio
async def test_parallel_turns_do_not_share_an_ack(monkeypatch):
    """The ContextVar must be per request, not a module-level list."""
    seen = []

    def _fake_run(text, context, hist):
        box = tools._ACK_BOX.get()
        if "first" in text:
            box.append("Включаю.")
        seen.append((text, box))
        return "готово"

    monkeypatch.setattr(worker_app, "_run_agent", _fake_run)
    req_a = worker_app.InvokeRequest(text="first", session_id="a", stream_name="kitchen")
    req_b = worker_app.InvokeRequest(text="second", session_id="b", stream_name="kitchen")

    async def _collect(req):
        return _events([c async for c in worker_app._handle(req)])

    results = await __import__("asyncio").gather(_collect(req_a), _collect(req_b))
    for events, expected in zip(results, (["Включаю.", "готово"], ["готово"])):
        sentences = [e["text"] for e in events if e.get("type") == "sentence"]
        assert sentences == expected, (
            f"acks leaked between turns: {sentences} != {expected}"
        )


@pytest.mark.asyncio
async def test_verified_ack_survives_a_worker_error(monkeypatch):
    """The action succeeded; a later model failure must not erase the ack."""
    def _fake_run(text, context, hist):
        tools._ACK_BOX.get().append("Включаю.")
        raise RuntimeError("model died after the tool")

    monkeypatch.setattr(worker_app, "_run_agent", _fake_run)
    req = worker_app.InvokeRequest(
        text="включи кофеварку", session_id="t", stream_name="kitchen"
    )
    events = _events([c async for c in worker_app._handle(req)])
    types = [e["type"] for e in events]
    assert types[:2] == ["sentence", "error"], (
        f"ack must be spoken before the error event, got {types}"
    )
    assert events[0]["text"] == "Включаю."


def test_partial_fanout_does_not_get_an_unqualified_ack(monkeypatch):
    payload = (
        '{"response_type":"action_done","data":{"success":['
        '{"type":"entity","id":"switch.a"},'
        '{"type":"entity","id":"switch.b"}],"failed":[]}}'
    )
    monkeypatch.setattr(
        tools, "confirm_on_off_verdict",
        lambda ids, target, fetch: (["switch.a"], ["switch.b"]),
    )
    # The first gate reads the live registry; stub it so the test reaches the
    # read-back branch instead of a 401 from the real HA.
    states = [
        {"entity_id": "switch.a", "state": "on", "attributes": {}},
        {"entity_id": "switch.b", "state": "off", "attributes": {}},
    ]
    monkeypatch.setattr(tools, "_raw_registry", lambda: (states, {}))
    box = []
    token = tools._ACK_BOX.set(box)
    try:
        out = tools._verify_on_off_result("intent__HassTurnOn", payload)
    finally:
        tools._ACK_BOX.reset(token)
    assert box == [], "a half-confirmed fan-out must not say a blanket «Включаю.»"
    assert "Подтверждение пока не пришло" in out or "частично" in out
