"""Tests for `voice_mcp.py` — the MCP (SSE) server Hermes uses for proactive
speech through the camera speakers.

The guards live in SERVER CODE, not in the Hermes skill prompt, so every
guard here is a unit test of the module, not of a prompt: quiet hours with
the overnight wrap and the MSK offset (the container runs UTC), the
per-room 120 s gap, the 240-char truncation, the named-and-known room rule,
and the Bearer wrapper's fail-closed behaviour.

`voice_mcp.py` imports only stdlib + loguru, so this file runs BOTH in the
Docker image and as a plain pytest run; `build_mcp_server()` (which needs
the `mcp` package) is deliberately not imported here.
"""

import asyncio
import calendar
import contextlib
import socket

import pytest

import voice_mcp


# --- fakes ------------------------------------------------------------------


class _StubSession:
    """Just enough of CameraSession: stream_name + _speak(text, reply)."""

    def __init__(self, name: str):
        self.stream_name = name
        self.spoken: list[str] = []

    async def _speak(self, text: str, reply: str = "") -> bool:
        self.spoken.append(text)
        return True


class _ChorusSession:
    """Enough of CameraSession for the chorus path: `_tts_fetch` + `_speak_pcm`.

    The chorus reuses `_speak_pcm` (not `_speak`) so every per-session guard —
    mic mute, echo ring, `_tts_play_end`, the ratio log — stays intact; a stub
    with only `_speak` cannot see the difference between "synthesised once"
    and "synthesised per room", which is the whole contract here.
    """

    def __init__(self, name: str):
        self.stream_name = name
        self.tts_calls: list[str] = []
        self.pcm_played: list[bytes] = []
        self.play_ok = True    # what the real _speak_pcm records as played

    async def _tts_fetch(self, text: str, speed: float = 1.0) -> bytes:
        self.tts_calls.append(text)
        return b"PCM"

    async def _speak_pcm(self, pcm: bytes, reply: str = "", wait=None) -> bool:
        self.pcm_played.append(pcm)
        # The real method records `played` here and returns `is_question` —
        # a property of the TEXT: False for a phrase that is not a question.
        # The chorus verdict MUST read the flag, not the return value — and
        # a stub that returned True here hid the second verdict bug exactly
        # the way the field showed it (17.11.2026: `r is not True` put every
        # successful non-question room into the failed list while all five
        # play_audio lines showed the audio played), so the stub returns the
        # real shape.
        self._last_play_ok = self.play_ok
        return "?" in (reply or "")


def _capture_loguru():
    """Collect loguru records as text; returns (list, remove-callback)."""
    msgs: list[str] = []
    handle = voice_mcp.logger.add(
        lambda m: msgs.append(str(m)), level="INFO")
    return msgs, lambda: voice_mcp.logger.remove(handle)


def _chorus_sessions() -> list[_ChorusSession]:
    return [_ChorusSession(n) for n in ("livingroom", "kitchen", "balcony")]


async def _drive(tool):
    """Run the tool AND its fire-and-forget dispatches on the SAME loop.

    The real deployment runs one loop, so a dispatch task is never cancelled;
    a second `asyncio.run` starts a NEW loop and the pending task is cancelled
    at its first await (measured: the chorus gather died there while the
    await-less single-room stub sailed through)."""
    res = await tool
    tasks = [t for t in list(voice_mcp._SPEAK_TASKS) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)
    return res


async def _drain_speak_tasks() -> None:
    """Let the fire-and-forget dispatches finish, then wait for them.

    `_speak_tool` creates its task and returns immediately (the real
    /play_audio POST must not block the tool call), so the stub's record is
    only real once the outstanding tasks are awaited.
    """
    tasks = [t for t in list(voice_mcp._SPEAK_TASKS) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


# --- the Bearer wrapper -----------------------------------------------------


def _run_auth(auth_header, token):
    """Drive BearerAuth with a hand-built ASGI scope; return (app_called, sent)."""
    app_calls: list = []
    sent: list = []

    async def app(scope, receive, send):
        app_calls.append(scope)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    headers = []
    if auth_header is not None:
        headers = [(b"authorization", auth_header.encode("latin-1"))]
    scope = {"type": "http", "method": "GET", "path": "/sse", "headers": headers}
    asyncio.run(voice_mcp.BearerAuth(app, token)(scope, receive, send))
    return app_calls, sent


def test_a_request_without_a_token_is_401_and_never_reaches_the_app():
    app_calls, sent = _run_auth(None, "secret-token")
    assert app_calls == []
    assert sent[0]["status"] == 401


def test_a_wrong_token_is_401():
    app_calls, sent = _run_auth("Bearer not-the-token", "secret-token")
    assert app_calls == []
    assert sent[0]["status"] == 401


def test_the_right_token_reaches_the_server():
    app_calls, sent = _run_auth("Bearer secret-token", "secret-token")
    assert len(app_calls) == 1
    assert not any(m.get("status") == 401 for m in sent)


def test_the_bearer_scheme_is_case_insensitive():
    # RFC 7235: auth schemes are case-insensitive; a strict comparison would
    # 401 a correct header sent as "bearer".
    app_calls, sent = _run_auth("bearer secret-token", "secret-token")
    assert len(app_calls) == 1


def test_an_empty_token_is_fail_closed_even_for_a_correct_header():
    # Empty MCP_TOKEN must disable EVERYTHING: no header variant gets in.
    app_calls, _ = _run_auth("Bearer ", "")
    assert app_calls == []


def test_a_lifespan_scope_passes_through_without_auth():
    # uvicorn sends the lifespan scope first; the wrapper must hand it to the
    # app or the server's own startup/shutdown never runs.
    app_calls: list = []

    async def app(scope, receive, send):
        app_calls.append(scope)

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(msg):
        pass

    asyncio.run(voice_mcp.BearerAuth(app, "")(
        {"type": "lifespan", "method": "GET", "headers": []}, receive, send))
    assert len(app_calls) == 1


# --- quiet hours ------------------------------------------------------------


def _epoch_utc(hour: int, minute: int = 0) -> float:
    return calendar.timegm((2026, 11, 16, hour, minute, 0))


def test_quiet_hours_covers_the_overnight_wrap_in_msk():
    # Window 23:00-08:00 MSK, container runs UTC: the wrap branch is the one
    # that matters — after 23:00 MSK AND before 08:00 MSK.
    assert voice_mcp.in_quiet_hours(_epoch_utc(20, 0)) is True    # 23:00 MSK
    assert voice_mcp.in_quiet_hours(_epoch_utc(21, 0)) is True    # 00:00 MSK
    assert voice_mcp.in_quiet_hours(_epoch_utc(2, 0)) is True     # 05:00 MSK
    assert voice_mcp.in_quiet_hours(_epoch_utc(4, 59)) is True    # 07:59 MSK
    assert voice_mcp.in_quiet_hours(_epoch_utc(5, 0)) is False    # 08:00 MSK
    assert voice_mcp.in_quiet_hours(_epoch_utc(12, 0)) is False   # 15:00 MSK


def test_the_offset_is_explicit_not_assumed():
    # With the MSK offset the same UTC clock is INSIDE the window; with 0 it
    # is not. This pins that the window is really shifted, not compared raw.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "PROACTIVE_QUIET_TZ_OFFSET_H", 0)
    try:
        assert voice_mcp.in_quiet_hours(_epoch_utc(20, 0)) is False
    finally:
        monkeypatch.undo()


def test_an_empty_window_disables_quiet_hours():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "PROACTIVE_QUIET_HOURS", "")
    try:
        assert voice_mcp.in_quiet_hours(_epoch_utc(21, 0)) is False
    finally:
        monkeypatch.undo()


def test_an_unparsable_window_falls_back_to_the_default():
    # The safe direction: garbage in the setting still keeps the night silent.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "PROACTIVE_QUIET_HOURS", "night")
    try:
        assert voice_mcp.in_quiet_hours(_epoch_utc(21, 0)) is True
    finally:
        monkeypatch.undo()


# --- the 240-char cap -------------------------------------------------------


def test_long_text_is_clipped_at_a_word_boundary():
    text = ("слово " * 60).strip()          # 360 chars
    clipped, was = voice_mcp.clip_text(text)
    assert was is True
    assert len(clipped) <= voice_mcp.PROACTIVE_MAX_CHARS
    assert clipped.endswith("слово")        # cut on a space, not mid-word


def test_short_text_passes_through():
    clipped, was = voice_mcp.clip_text("Привет!")
    assert (clipped, was) == ("Привет!", False)


def test_a_zero_cap_disables_the_cap():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "PROACTIVE_MAX_CHARS", 0)
    try:
        text = "слово " * 60
        clipped, was = voice_mcp.clip_text(text)
        assert was is False and clipped == text
    finally:
        monkeypatch.undo()


# --- the per-room gap -------------------------------------------------------


def test_the_second_phrase_within_the_gap_is_refused():
    base = _epoch_utc(12)                    # 15:00 MSK — not quiet hours
    last_at = {"livingroom": base}
    reason = voice_mcp.check_speak_allowed("text", "livingroom", base + 60,
                                           last_at=last_at)
    assert "min gap" in reason
    assert "retry in" in reason


def test_the_gap_is_per_room_not_global():
    base = _epoch_utc(12)
    last_at = {"livingroom": base}
    assert voice_mcp.check_speak_allowed("text", "kitchen", base + 60,
                                         last_at=last_at) == ""


def test_after_the_gap_the_room_speaks_again():
    base = _epoch_utc(12)
    last_at = {"livingroom": base}
    assert voice_mcp.check_speak_allowed(
        "text", "livingroom", base + voice_mcp.PROACTIVE_MIN_GAP_S,
        last_at=last_at) == ""


def test_a_zero_gap_disables_the_gap():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "PROACTIVE_MIN_GAP_S", 0.0)
    try:
        base = _epoch_utc(12)
        last_at = {"livingroom": base}
        assert voice_mcp.check_speak_allowed("text", "livingroom", base + 1,
                                             last_at=last_at) == ""
    finally:
        monkeypatch.undo()


# --- the tool itself --------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Every test starts with its own session list and an empty gap table."""
    monkeypatch.setattr(voice_mcp, "_SESSIONS", [])
    monkeypatch.setattr(voice_mcp, "_LAST_SPEAK_AT", {})
    yield


def test_speak_dispatches_to_the_named_room():
    s = _StubSession("livingroom")
    voice_mcp._SESSIONS.append(s)
    res = asyncio.run(voice_mcp._speak_tool("Тест", "livingroom",
                                            now=_epoch_utc(12)))
    asyncio.run(_drain_speak_tasks())
    assert '"spoken": true' in res or '"spoken":True' in res.replace(" ", "")
    assert s.spoken == ["Тест"]
    assert voice_mcp._LAST_SPEAK_AT["livingroom"] == _epoch_utc(12)


def test_an_empty_room_is_refused_with_the_rooms_list():
    voice_mcp._SESSIONS.append(_StubSession("livingroom"))
    res = asyncio.run(voice_mcp._speak_tool("Тест", "", now=1000.0))
    assert '"rooms"' in res and "livingroom" in res
    assert "spoken" not in res


def test_an_unknown_room_is_refused_and_names_the_rooms():
    voice_mcp._SESSIONS.append(_StubSession("livingroom"))
    res = asyncio.run(voice_mcp._speak_tool("Тест", " attic", now=1000.0))
    assert "unknown room" in res
    assert "livingroom" in res           # the model can recover by reading it


def test_quiet_hours_refuse_the_tool_and_dispatch_nothing():
    s = _StubSession("livingroom")
    voice_mcp._SESSIONS.append(s)
    # 21:00 UTC = 00:00 MSK — inside the shipped quiet window.
    res = asyncio.run(voice_mcp._speak_tool(
        "Тест", "livingroom", now=_epoch_utc(21)))
    assert "quiet hours" in res
    assert s.spoken == []
    assert "livingroom" not in voice_mcp._LAST_SPEAK_AT


def test_the_gap_refuses_the_tool_before_dispatch():
    s = _StubSession("livingroom")
    voice_mcp._SESSIONS.append(s)
    base = _epoch_utc(12)
    asyncio.run(voice_mcp._speak_tool("Тест", "livingroom", now=base))
    asyncio.run(_drain_speak_tasks())
    assert s.spoken == ["Тест"]
    res = asyncio.run(voice_mcp._speak_tool("Ещё", "livingroom", now=base + 30))
    assert "min gap" in res
    assert s.spoken == ["Тест"]          # nothing was dispatched twice


def test_long_tool_text_is_truncated_and_said_shorter():
    s = _StubSession("livingroom")
    voice_mcp._SESSIONS.append(s)
    text = "слово " * 60
    res = asyncio.run(voice_mcp._speak_tool(text, "livingroom",
                                            now=_epoch_utc(12)))
    asyncio.run(_drain_speak_tasks())
    assert "truncated" in res
    assert s.spoken and len(s.spoken[0]) <= voice_mcp.PROACTIVE_MAX_CHARS


def test_rooms_lists_the_live_session_names_once_each():
    voice_mcp._SESSIONS.extend([_StubSession("kitchen"), _StubSession("kitchen"),
                                _StubSession("livingroom")])
    names = voice_mcp._room_names()
    assert names == ["kitchen", "livingroom"]


def test_disabled_server_reasons():
    # Each case keeps MCP_TOKEN patched, because undo() restores the MODULE's
    # real value — "" in the test image, where MCP_TOKEN is not set — and an
    # empty token reports itself before the port ever gets a say.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "MCP_TOKEN", "")
    try:
        assert "MCP_TOKEN" in voice_mcp._mcp_disabled_reason()
    finally:
        monkeypatch.undo()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(voice_mcp, "MCP_TOKEN", "tok")
    monkeypatch.setattr(voice_mcp, "PROACTIVE_MCP_PORT", 0)
    try:
        assert "PROACTIVE_MCP_PORT" in voice_mcp._mcp_disabled_reason()
        monkeypatch.setattr(voice_mcp, "PROACTIVE_MCP_PORT", 18793)
        assert voice_mcp._mcp_disabled_reason() == ""
    finally:
        monkeypatch.undo()


# --- the chorus (room="all") -------------------------------------------------


def test_the_chorus_synthesises_once_and_plays_the_same_bytes_everywhere():
    """STRICT chorus (the user's «строго хором одновременно»): one synthesis,
    the very same bytes object handed to every room, all POSTs started
    together. A per-room `_tts_fetch` would make three rooms that speak
    seconds apart — not a chorus.
    """
    sessions = _chorus_sessions()
    voice_mcp._SESSIONS.extend(sessions)
    base = _epoch_utc(12)
    res = asyncio.run(_drive(voice_mcp._speak_tool("Хором", "all", now=base)))
    assert '"spoken": true' in res
    assert '"room": "all"' in res
    # ONE synthesis through sessions[0], not one per room
    assert [s.tts_calls for s in sessions] == [["Хором"], [], []]
    # the SAME bytes object in every room — identity, not equality
    assert [s.pcm_played for s in sessions] == [[b"PCM"]] * 3
    assert sessions[0].pcm_played[0] is sessions[1].pcm_played[0]
    assert sessions[0].pcm_played[0] is sessions[2].pcm_played[0]
    # every room's gap timer is set, so the NEXT chorus is also gap-checked
    for s in sessions:
        assert voice_mcp._LAST_SPEAK_AT[s.stream_name] == base


def test_one_room_inside_its_gap_refuses_the_whole_chorus():
    """ALL-OR-NOTHING: a room inside its own 120 s gap would make a partial
    chorus («камера промолчала»), so the whole call is refused and the
    blocked rooms are named — nothing is dispatched anywhere.
    """
    sessions = _chorus_sessions()
    voice_mcp._SESSIONS.extend(sessions)
    base = _epoch_utc(12)
    voice_mcp._LAST_SPEAK_AT["kitchen"] = base
    res = asyncio.run(voice_mcp._speak_tool("Хором", "all", now=base + 30))
    assert "chorus needs every room free" in res
    assert "blocked" in res and "kitchen" in res
    assert "spoken" not in res
    assert all(s.pcm_played == [] for s in sessions)


def test_quiet_hours_refuse_the_chorus_and_dispatch_nothing():
    sessions = _chorus_sessions()
    voice_mcp._SESSIONS.extend(sessions)
    # 21:00 UTC = 00:00 MSK — inside the shipped quiet window.
    res = asyncio.run(voice_mcp._speak_tool("Хором", "all", now=_epoch_utc(21)))
    assert "quiet hours" in res
    assert all(s.pcm_played == [] for s in sessions)
    assert voice_mcp._LAST_SPEAK_AT == {}


def test_the_chorus_with_no_sessions_is_refused():
    res = asyncio.run(voice_mcp._speak_tool("Хором", "all", now=_epoch_utc(12)))
    assert "no camera sessions" in res
    assert "spoken" not in res


def test_a_failed_synthesis_leaves_every_room_silent():
    """The tool answer is fire-and-forget like the single-room path, so a dead
    TTS still answers spoken:true — the honest trace is the log line, and no
    room hears a half broadcast."""
    sessions = _chorus_sessions()
    voice_mcp._SESSIONS.extend(sessions)

    async def _empty(text: str, speed: float = 1.0) -> bytes | None:
        return None

    sessions[0]._tts_fetch = _empty
    res = asyncio.run(_drive(voice_mcp._speak_tool("Хором", "all",
                                                   now=_epoch_utc(12))))
    assert '"spoken": true' in res
    assert all(s.pcm_played == [] for s in sessions)


def test_the_chorus_still_truncates_long_text():
    sessions = _chorus_sessions()
    voice_mcp._SESSIONS.extend(sessions)
    text = "слово " * 60
    res = asyncio.run(_drive(voice_mcp._speak_tool(text, "all",
                                                   now=_epoch_utc(12))))
    assert "truncated" in res
    assert sessions[0].tts_calls == [
        text[:voice_mcp.PROACTIVE_MAX_CHARS].rsplit(" ", 1)[0].strip()
    ]


def test_the_chorus_verdict_names_who_actually_played():
    """`_speak_pcm` returns `is_question` — a property of the TEXT (documented
    10.10.2026), so a verdict built on the return value called a room that
    PLAYED a failure and stayed silent about one that timed out. The honest
    verdict reads `_last_play_ok`."""
    sessions = _chorus_sessions()
    sessions[1].play_ok = False          # kitchen: the POST did not play
    voice_mcp._SESSIONS.extend(sessions)
    msgs, remove = _capture_loguru()
    try:
        asyncio.run(_drive(voice_mcp._speak_tool("Хором", "all",
                                                 now=_epoch_utc(12))))
        line = " ".join(msgs)
        assert "PROACTIVE chorus: spoken in ['livingroom', 'balcony']" in line
        assert "failed: [('kitchen', 'False')]" in line
    finally:
        remove()


def test_a_raised_playback_failure_still_names_the_room():
    """The exception branch: under return_exceptions=True a room whose
    `_speak_pcm` raises comes back as an exception INSTANCE — `isinstance` is
    what catches it. The old `r is not True` could not distinguish "raised"
    from "returned False" (17.11.2026 probe #2: all five successful rooms were
    reported failed because the real method returns is_question=False)."""
    sessions = _chorus_sessions()

    async def boom(pcm, reply="", wait=None):
        raise RuntimeError("camera POST refused")

    sessions[1]._speak_pcm = boom                 # kitchen raises
    voice_mcp._SESSIONS.extend(sessions)
    msgs, remove = _capture_loguru()
    try:
        asyncio.run(_drive(voice_mcp._speak_tool("Хором", "all",
                                                 now=_epoch_utc(12))))
        line = " ".join(msgs)
        assert "PROACTIVE chorus: spoken in ['livingroom', 'balcony']" in line
        # repr(r) is a STRING containing single quotes, so the tuple's own
        # repr wraps it in double quotes.
        assert ('failed: [(\'kitchen\', "RuntimeError(\'camera POST '
                'refused\')")]') in line
    finally:
        remove()


async def _sse_probe(port: int, token: str | None, host: str = "127.0.0.1") -> str:
    """GET /sse once over a raw socket; return '401', '200+event' or an error.

    Raw asyncio, not httpx: the authed reply is an SSE STREAM that never ends,
    and the test must not block the loop the server task is running on.
    """
    try:
        r, w = await asyncio.open_connection("127.0.0.1", port)
    except Exception as e:
        return f"connect refused: {e!r}"
    try:
        req = b"GET /sse HTTP/1.1\r\n"
        req += f"Host: {host}:{port}\r\n".encode()   # the SSE transport
        req += b"Accept: text/event-stream\r\n"      # VALIDATES the Host
        if token:
            req += f"Authorization: Bearer {token}\r\n".encode()
        req += b"\r\n"
        w.write(req)
        await w.drain()
        status = await asyncio.wait_for(r.readline(), timeout=5)
        ctype = b""
        if b"200" in status:
            for _ in range(30):
                line = await asyncio.wait_for(r.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break
                if line.startswith(b"content-type:"):
                    ctype = line
        w.close()
        if b"401" in status:
            return "401"
        if b"200" in status and b"text/event-stream" in ctype:
            return "200+event"
        return f"unexpected {status!r}"
    except Exception as e:
        return f"probe error: {e!r}"
    finally:
        w.close()


def test_the_server_task_binds_sse_and_closes_cleanly():
    """The REAL run_mcp_server() task: the port opens, 401 fail-closed, SSE
    with the token, and a bounded shutdown on cancel.

    This pins the uvicorn lifespan wiring. A hand-driven server without
    `server.lifespan = config.lifespan_class(config)` dies at uvicorn's
    `await self.lifespan.startup()` — startup() never creates the lifespan,
    serve() does — and the port simply never opens (measured 16.11.2026:
    the gateway logged the starting line and :18793 stayed dead, with no
    exception anywhere in the log).
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    old = (voice_mcp.MCP_TOKEN, voice_mcp.PROACTIVE_MCP_PORT)
    voice_mcp.MCP_TOKEN = "test-token"
    voice_mcp.PROACTIVE_MCP_PORT = port
    try:

        async def drive():
            task = asyncio.create_task(voice_mcp.run_mcp_server(),
                                       name="voice-mcp-test")
            for _ in range(50):
                await asyncio.sleep(0.1)
                if task.done():
                    pytest.fail(
                        f"server task died before opening: {task.exception()!r}")
                try:
                    r, w = await asyncio.open_connection("127.0.0.1", port)
                    w.close()
                    break
                except Exception:
                    continue
            else:
                task.cancel()
                pytest.fail("the server port never opened in 5 s")

            anon = await _sse_probe(port, None)
            auth = await _sse_probe(port, "test-token")
            lan = await _sse_probe(port, "test-token", host="192.168.22.102")

            task.cancel()
            # A bounded shutdown (timeout_graceful_shutdown) must let the task
            # finish even with an SSE client — a hung teardown would hang the
            # gateway's own shutdown behind it.
            with contextlib.suppress(asyncio.CancelledError,
                                     asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=10)
            assert task.done(), "server task did not finish after cancel"

            assert anon == "401", anon           # fail-closed without a token
            assert auth == "200+event", auth     # SSE opens with it
            # The path hermes actually takes: the client connects to the
            # gateway's LAN IP, so the Host header is `192.168.22.102:<port>`.
            # With the FastMCP DEFAULTS the SDK's DNS-rebinding guard
            # allowlists only localhost and answered 421 Misdirected Request
            # to every real client (measured 17.11.2026 on the live gateway).
            assert lan == "200+event", lan

        asyncio.run(drive())
    finally:
        voice_mcp.MCP_TOKEN, voice_mcp.PROACTIVE_MCP_PORT = old
