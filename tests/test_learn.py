"""Self-learning loop: a lesson is applied only after an upper-level success."""

import asyncio
import importlib.util
import json
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import lessons  # noqa: E402
import memory  # noqa: E402


def _router_app():
    spec = importlib.util.spec_from_file_location(
        "jev_router_learn_app2", os.path.join(_ROUTER, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate(**kw):
    base = {
        "lesson_id": "abc",
        "from_text": "включи кафеварку",
        "to_text": "включи кофеварку",
        "outcome": {"kind": "ha_action", "confirmed": True},
        "source_level": "l2",
        "confidence": 0.9,
    }
    base.update(kw)
    return base


def test_rewrite_is_exact_and_leaves_other_text_alone():
    lessons._ACTIVE[:] = [_candidate()]
    assert lessons.rewrite("включи кафеварку")[0] == "включи кофеварку"
    assert lessons.rewrite("выключи кафеварку")[0] == "выключи кафеварку"
    assert lessons.rewrite("включи кофеварку")[1] is None
    lessons._ACTIVE[:] = []


def test_validate_refuses_code_and_tool_names():
    for bad in (
        _candidate(to_text="import os; ha_action(x)"),
        _candidate(to_text="__import__('os')"),
        _candidate(to_text="включи свет", from_text="x" * 200),
        _candidate(to_text="http://192.168.22.102:8091/route"),
    ):
        assert lessons.validate(bad, [])[0] is False, bad


def test_validate_refuses_a_rewrite_that_changes_an_already_successful_turn():
    history = [
        {"text": "включи кафеварку", "route": "easy_action"},
        {"text": "сколько весит", "route": "scale"},
    ]
    ok, why = lessons.validate(_candidate(), history)
    assert ok is False and "history" in why


def test_validate_accepts_a_safe_synonym():
    ok, why = lessons.validate(
        _candidate(),
        [{"text": "включи кофеварку", "route": "easy_action"}],
    )
    assert ok is True, why


def test_commit_requires_confirmed_outcome_and_no_error_repeat_or_veto():
    signals = {
        "error": False, "veto": False, "sentence": True,
        "done": True, "repeated": False, "outcome": {"confirmed": True},
    }
    assert lessons.should_commit(signals)[0] is True
    for field, value in (
        ("error", True), ("veto", True), ("sentence", False),
        ("done", False), ("repeated", True),
    ):
        bad = dict(signals); bad[field] = value
        assert lessons.should_commit(bad)[0] is False, field
    bad = dict(signals); bad["outcome"] = {"confirmed": False}
    assert lessons.should_commit(bad)[0] is False


@pytest.mark.asyncio
async def test_a_lesson_is_committed_only_after_a_clean_answer(monkeypatch):
    """The worker's candidate must be held until the router sees a real answer,
    and validated against already-successful turns before it becomes active."""
    app = _router_app()
    statuses = []

    async def _status(lesson_id, status, reason=""):
        statuses.append((status, reason))

    async def _history(*a, **k):
        return []

    async def _classify(*a, **k):
        return type("D", (), {"route": "complex_logic", "confidence": 0.5,
                              "reason": "probe"})()

    candidate = {
        "type": "lesson", "from_text": "включи кафеварку",
        "to_text": "включи кофеварку", "source_level": "l2",
        "outcome": {"kind": "stt_repair", "confirmed": True},
    }

    async def _worker(*a, **k):
        out = k.get("lessons_out")
        yield {"type": "sentence", "text": "Включила кофеварку."}
        if out is not None:
            out.append(candidate)
        yield {"type": "done", "elapsed": 1.0}

    monkeypatch.setattr(app.lessons, "set_status", _status)
    monkeypatch.setattr(app.lessons, "successful_history", _history)
    monkeypatch.setattr(app.classifier, "classify", _classify)
    monkeypatch.setattr(app, "_run_worker", _worker)

    async def _drive():
        return [
            json.loads(c.removeprefix("data: "))
            async for c in app._handle(app.RouteRequest(
                text="включи кафеварку", session_id="probe", stream_name="kitchen"
            ))
        ]

    events = await _drive()
    assert any(e.get("type") == "sentence" for e in events), events
    assert statuses and statuses[0][0] == "active", statuses


def test_history_selection_uses_recent_deterministic_turns_only():
    """The validation corpus must not be 59 days of mixed escalations:
    a rule that fixes today's phrasing must not be vetoed by an old L2 turn."""
    now = 1_800_000_000.0
    points = [
        {"payload": {"text": "a", "reply": "ok", "route": "easy_action",
                     "stream_name": "kitchen", "ts": now - 100}},
        {"payload": {"text": "b", "reply": "ok", "route": "complex_logic",
                     "stream_name": "kitchen", "ts": now - 200}},
        {"payload": {"text": "c", "reply": "ok", "route": "easy_query",
                     "stream_name": "kitchen", "ts": now - 9 * 86400}},
        {"payload": {"text": "d", "reply": "", "route": "easy_action",
                     "stream_name": "kitchen", "ts": now - 300}},
    ]
    selected = lessons.select_history(points, now=now)
    assert [r["text"] for r in selected] == ["a"]


def test_history_selection_caps_each_room():
    now = 1_800_000_000.0
    points = [
        {"payload": {"text": f"k{i}", "reply": "ok", "route": "easy_action",
                     "stream_name": "kitchen", "ts": now - i}}
        for i in range(20)
    ]
    assert len(lessons.select_history(points, now=now, per_room=8, total=60)) == 8


def test_a_failed_lesson_is_never_reused():
    lessons._ACTIVE[:] = [_candidate(), _candidate(lesson_id="ok")]
    lessons.mark_failed("abc", "escalated again")
    assert [r["lesson_id"] for r in lessons._ACTIVE] == ["ok"]
    lessons._ACTIVE[:] = []


def test_router_applies_a_learned_rewrite_before_the_classifier(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "jev_router_learn_app", os.path.join(_ROUTER, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.lessons, "_ACTIVE", [_candidate()], raising=False)
    monkeypatch.setattr(
        mod.lessons, "rewrite",
        lambda t: ("включи кофеварку", "abc") if "кафеварку" in t else (t, None),
    )

    seen = []

    async def _classify(text, *a, **k):
        seen.append(text)
        return type("D", (), {"route": "easy_query", "confidence": 0.9,
                              "reason": ""})()

    monkeypatch.setattr(mod.classifier, "classify", _classify)

    async def _drive():
        gen = mod._handle(mod.RouteRequest(
            text="включи кафеварку", session_id="probe", stream_name="kitchen"
        ))
        first = await gen.__anext__()
        await gen.aclose()
        return json.loads(first.removeprefix("data: "))

    route = __import__("asyncio").run(_drive())
    assert route["type"] == "route"
    assert seen == ["включи кофеварку"], seen


async def _drive_events(app, text, stream="kitchen"):
    """Drive a whole _handle turn and return the decoded SSE events."""
    return [
        json.loads(c.removeprefix("data: "))
        async for c in app._handle(app.RouteRequest(
            text=text, session_id="probe", stream_name=stream
        ))
    ]


def _stub_complex_classifier(monkeypatch, app):
    async def _classify(*a, **k):
        return type("D", (), {"route": "complex_logic", "confidence": 0.5,
                              "reason": "probe"})()
    monkeypatch.setattr(app.classifier, "classify", _classify)
    async def _history(*a, **k):
        return []
    monkeypatch.setattr(app.lessons, "successful_history", _history)


@pytest.mark.asyncio
async def test_a_repeat_quarantines_the_lesson_that_was_applied(monkeypatch):
    """A turn that merely APPLIED an existing lesson wrote an empty id into
    `_RECENT`, so `is_repeat()` came back empty and the applied rule could never
    be quarantined when the user had to say the same thing twice. The id that
    was actually used (rewrite's second value) must be the one recorded.
    """
    app = _router_app()
    lessons._RECENT.clear()
    statuses = []

    async def _status(lesson_id, status, reason=""):
        statuses.append((lesson_id, status, reason))

    # Production rewrite is synchronous; it returns (new text, applied id).
    monkeypatch.setattr(
        app.lessons, "rewrite",
        lambda t: ("включи кофеварку", "lesson-L") if "кафеварку" in t else (t, None),
    )
    monkeypatch.setattr(app.lessons, "set_status", _status)
    _stub_complex_classifier(monkeypatch, app)

    async def _worker(*a, **k):
        yield {"type": "sentence", "text": "Включила кофеварку."}

    monkeypatch.setattr(app, "_run_worker", _worker)

    raw = "включи кафеварку"
    await _drive_events(app, raw)
    await asyncio.sleep(0)
    # The applied id, not "", must be what the repeat detector reads back.
    assert lessons._RECENT[("kitchen", raw)][1] == "lesson-L"
    assert statuses == [], statuses  # the first turn is not a repeat

    await _drive_events(app, raw)
    await asyncio.sleep(0)
    assert ("lesson-L", "failed") in [(s[0], s[1]) for s in statuses], statuses
    lessons._RECENT.clear()


@pytest.mark.asyncio
async def test_a_fresh_commit_records_its_own_id_not_an_applied_one(monkeypatch):
    """Guard: when THIS turn committed a new rule, its own id wins over any
    id that an earlier rewrite might have applied. A fresh commit must still be
    remembered so a later repeat quarantines the rule it actually created.
    """
    app = _router_app()
    lessons._RECENT.clear()
    committed = {}

    async def _status(lesson_id, status, reason=""):
        committed["id"], committed["status"] = lesson_id, status

    async def _worker(*a, **k):
        out = k.get("lessons_out")
        yield {"type": "sentence", "text": "Включила кофеварку."}
        if out is not None:
            out.append({
                "type": "lesson", "from_text": "включи кафеварку",
                "to_text": "включи кофеварку", "source_level": "l2",
                "outcome": {"kind": "stt_repair", "confirmed": True},
            })
        yield {"type": "done", "elapsed": 1.0}

    monkeypatch.setattr(app.lessons, "set_status", _status)
    _stub_complex_classifier(monkeypatch, app)
    monkeypatch.setattr(app, "_run_worker", _worker)

    raw = "включи кафеварку"
    await _drive_events(app, raw)
    assert committed.get("status") == "active", committed
    assert lessons._RECENT[("kitchen", raw)][1] == committed["id"], (
        lessons._RECENT[("kitchen", raw)], committed
    )
    lessons._RECENT.clear()


@pytest.mark.asyncio
async def test_a_committed_lesson_is_persisted_before_it_is_activated(monkeypatch):
    """`lessons.save()` is the only writer of the pending point, and it was
    never called: `set_status("active")` only PATCHes an existing point, so a
    committed rule existed nowhere but the in-process cache. Persist it first.
    """
    app = _router_app()
    lessons._ACTIVE[:] = []
    lessons._RECENT.clear()
    calls = []
    saved = {}

    async def _embed(text):
        return [0.01] * 4

    async def _request(method, path, payload=None):
        calls.append((method, path))
        if method == "PUT" and path.startswith("/collections/voice_lessons/points"):
            saved["point"] = (payload or {}).get("points", [{}])[0]
            return 200, {}
        if method == "POST" and "/collections/voice_lessons/points/payload" in path:
            return 200, {}
        if method == "POST" and "/collections/voice_lessons/points/scroll" in path:
            pl = dict(saved.get("point", {}).get("payload", {}))
            pl["status"] = "active"
            return 200, {"result": {"points": [{"payload": pl}]}}
        return 200, {}

    async def _worker(*a, **k):
        out = k.get("lessons_out")
        yield {"type": "sentence", "text": "Включила кофеварку."}
        if out is not None:
            out.append({
                "type": "lesson", "from_text": "включи кафеварку",
                "to_text": "включи кофеварку", "source_level": "l2",
                "outcome": {"kind": "stt_repair", "confirmed": True},
            })
        yield {"type": "done", "elapsed": 1.0}

    monkeypatch.setattr(memory, "_request", _request)
    monkeypatch.setattr(memory, "_embed", _embed)
    monkeypatch.setattr(app.lessons, "successful_history", _history_noop)
    _stub_complex_classifier(monkeypatch, app)
    monkeypatch.setattr(app, "_run_worker", _worker)

    await _drive_events(app, "включи кафеварку")

    puts = [i for i, c in enumerate(calls)
            if c[0] == "PUT" and c[1].startswith("/collections/voice_lessons/points")]
    patches = [i for i, c in enumerate(calls)
               if c[0] == "POST" and "/collections/voice_lessons/points/payload" in c[1]]
    assert puts, "lessons.save() was never called — no pending point exists"
    assert patches, "set_status() never patched the point"
    assert puts[0] < patches[0], "the point must exist before its status is patched"
    lid = saved["point"]["payload"]["lesson_id"]
    rewritten, applied = lessons.rewrite("включи кафеварку")
    assert rewritten == "включи кофеварку" and applied == lid, (rewritten, applied)
    lessons._ACTIVE[:] = []
    lessons._RECENT.clear()


async def _history_noop(*a, **k):
    return []


@pytest.mark.asyncio
async def test_save_never_raises_on_transport_failure(monkeypatch):
    """Guard: the new call sits in the live turn path, so save() must keep its
    never-raise contract rather than turning a Qdrant outage into a dead turn.
    """
    async def _boom(*a, **k):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(memory, "_embed", _boom)
    await lessons.save(_candidate())  # must not raise


@pytest.mark.asyncio
async def test_a_lesson_from_a_crashed_worker_is_rejected(monkeypatch):
    """A worker that yields a sentence AND a lesson and then times out emitted
    an error, so the turn was not a clean success. The signals were hardcoded
    (`error=False`, `done=True`) and the rule was committed anyway.
    """
    app = _router_app()
    lessons._ACTIVE[:] = []
    lessons._RECENT.clear()
    statuses = []
    seen = []
    real_should_commit = app.lessons.should_commit

    async def _status(lesson_id, status, reason=""):
        statuses.append((status, reason))

    def _should_commit(signals):
        seen.append(dict(signals))
        return real_should_commit(signals)

    async def _worker(*a, **k):
        out = k.get("lessons_out")
        yield {"type": "sentence", "text": "Включила кофеварку."}
        if out is not None:
            out.append({
                "type": "lesson", "from_text": "включи кафеварку",
                "to_text": "включи кофеварку", "source_level": "l2",
                "outcome": {"kind": "stt_repair", "confirmed": True},
            })
        raise RuntimeError("worker timed out after its sentence")

    monkeypatch.setattr(app.lessons, "set_status", _status)
    monkeypatch.setattr(app.lessons, "should_commit", _should_commit)
    monkeypatch.setattr(app.lessons, "successful_history", _history_noop)
    _stub_complex_classifier(monkeypatch, app)
    monkeypatch.setattr(app, "_run_worker", _worker)

    events = await _drive_events(app, "включи кафеварку")
    assert any(e.get("type") == "error" for e in events), events
    assert seen and seen[0]["error"] is True, seen
    assert seen[0]["done"] is False, seen
    assert ("rejected", "error event") in statuses, statuses
    assert not any(s[0] == "active" for s in statuses), statuses
    assert not any(r.get("lesson_id") for r in lessons._ACTIVE), lessons._ACTIVE
    lessons._RECENT.clear()
