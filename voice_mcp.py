"""MCP (SSE) server exposing `speak(text, room)` and `rooms()` to Hermes.

WHAT IT IS FOR: Hermes (hermes-agent) asks a camera room to speak one short
phrase on its OWN initiative — a proactive announcement. It is never a reply
to a user command: replies keep the normal dialogue path.

WHY A SEPARATE PORT, NOT A MOUNT: main.py already owns POST /mcp/{session_id}
(the ESP32 MCP JSON-RPC passthrough). A Starlette mount under /mcp registers
AFTER that route, so the MCP client's POST /messages/?session_id=... would
match the ESP32 route first and reach the wrong handler silently. FastMCP's
sse_app() is therefore served on its own port (PROACTIVE_MCP_PORT, default
18793); the gateway runs with network_mode=host, so no ports: entry is
needed in compose.

WHY `mcp<2` (pinned in requirements.txt): mcp 2.x renamed FastMCP to
MCPServer (`mcp.server.mcpserver`), and v1's `mcp.server.fastmcp.FastMCP`
is gone. This module is written against the v1 API; the SDK itself
recommends pinning for v1 code (verified live in hermes' venv, 16.11.2026).

AUTH: `Authorization: Bearer <MCP_TOKEN>` on EVERY request — GET /sse and
POST /messages/ both. An empty MCP_TOKEN disables the server entirely
(fail-closed, the Telegram convention). This token is NOT the gateway's
admin Basic creds; Hermes never holds those.

GUARDS, IN CODE AND NOT IN A PROMPT (the model cannot be trusted to enforce
its own limits):
  * PROACTIVE_QUIET_HOURS (default "23:00-08:00") — no proactive speech at
    night. The hamster cage sits under a camera speaker; a 3 a.m.
    announcement is the failure mode this exists to prevent.
  * PROACTIVE_MIN_GAP_S (default 120) per room.
  * PROACTIVE_MAX_CHARS (default 240) — longer text is TRUNCATED at a word
    boundary, not refused: a truncated announcement still carries its
    point, and the tool result says so, which is what teaches the model to
    stay short. `0` disables the cap (the "0 means unset" convention).
  * the room must be NAMED and must EXIST; "" and unknown rooms are
    refused with the live rooms list, never guessed.
Timezone: the gateway container runs UTC (measured: `date` prints UTC,
16.11.2026) while the household runs MSK, so the quiet window is compared
against UTC+PROACTIVE_QUIET_TZ_OFFSET_H (default 3 = MSK).

The guards run on the SERVER (this module), not in the Hermes skill: the
skill only documents them.

Module-level conventions, same as the rest of this repo: env read at
import, tables and regexes at module level (the `_TIME_RE` class-body
lesson), and no `import main` anywhere — main passes the LIVE session list
in through set_sessions() instead, the telegram_client injection pattern.
"""

import asyncio
import hmac
import json
import os
import re
import time

from loguru import logger

# --- env (module level, like every other module in this repo) ---------------

# Empty = the whole MCP server is disabled, fail-closed (Telegram convention).
MCP_TOKEN = os.getenv("MCP_TOKEN", "")

# Own port: see the module docstring for why this cannot be a mount.
# `0` disables the server without taking the token away.
PROACTIVE_MCP_PORT = int(os.getenv("PROACTIVE_MCP_PORT", "18793") or 0)

# Quiet window in "HH:MM-HH:MM" form, compared against UTC+_QUIET_TZ_OFFSET_H.
# Empty string disables quiet hours entirely (daytime testing).
PROACTIVE_QUIET_HOURS = os.getenv("PROACTIVE_QUIET_HOURS", "23:00-08:00")

# The container runs UTC; the household runs MSK. Explicit 0 means UTC
# (not "unset" — an offset of zero is a real choice here).
PROACTIVE_QUIET_TZ_OFFSET_H = int(os.getenv("PROACTIVE_QUIET_TZ_OFFSET_H", "3"))

PROACTIVE_MIN_GAP_S = float(os.getenv("PROACTIVE_MIN_GAP_S", "120") or 0)
PROACTIVE_MAX_CHARS = int(os.getenv("PROACTIVE_MAX_CHARS", "240") or 0)

_DEFAULT_QUIET_WINDOW = (23 * 60, 8 * 60)   # the shipped default, MSK
_QUIET_WINDOW_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$")

# --- state ------------------------------------------------------------------

# The LIVE camera session list, handed over by main.on_startup via
# set_sessions(). Held by reference: start_camera_sessions() appends to the
# very same list, so rooms() always reports what is actually running.
_SESSIONS: list = []


def set_sessions(sessions: list) -> None:
    """Point the MCP tools at the live session list (main.py's own list)."""
    global _SESSIONS
    _SESSIONS = sessions


# Last proactive speech per room, for the min-gap guard. Keyed by ROOM, not
# globally: two rooms are independent speakers.
_LAST_SPEAK_AT: dict[str, float] = {}

# Outstanding playback tasks, so a test can await them and shutdown can
# cancel them (a proactive phrase outliving the gateway stop is noise).
_SPEAK_TASKS: set = set()


def _room_names() -> list[str]:
    seen: list[str] = []
    for s in _SESSIONS:
        name = getattr(s, "stream_name", "")
        if name and name not in seen:
            seen.append(name)
    return seen


# --- guards (pure functions; testable without asyncio or a server) ----------


def parse_quiet_window(spec: str):
    """`"23:00-08:00"` -> `(1380, 480)`; None when absent or unparsable."""
    m = _QUIET_WINDOW_RE.match((spec or "").strip())
    if not m:
        return None
    start_h, start_m, end_h, end_m = (int(g) for g in m.groups())
    if start_h > 23 or end_h > 23 or start_m > 59 or end_m > 59:
        return None
    return start_h * 60 + start_m, end_h * 60 + end_m


_quiet_warned = False


def in_quiet_hours(now: float) -> bool:
    """True when `now` (epoch seconds) sits inside the quiet window.

    The window is expressed in UTC+PROACTIVE_QUIET_TZ_OFFSET_H, because the
    container's own clock is UTC while the household is MSK — comparing
    against raw UTC would move the window 3 h into the morning and have
    spoken into the night it was written to keep silent.
    """
    global _quiet_warned
    spec = (PROACTIVE_QUIET_HOURS or "").strip()
    if not spec:
        return False
    win = parse_quiet_window(spec)
    if win is None:
        # Unparsable setting falls back to the shipped default, the safe
        # direction: refusing to speak can only cost a message, while
        # speaking at 3 a.m. is the fault this guard exists to prevent.
        if not _quiet_warned:
            logger.warning(
                f"PROACTIVE_QUIET_HOURS {spec!r} unparsable — "
                f"falling back to 23:00-08:00"
            )
            _quiet_warned = True
        win = _DEFAULT_QUIET_WINDOW
    start, end = win
    t = time.gmtime(now + PROACTIVE_QUIET_TZ_OFFSET_H * 3600)
    mins = t.tm_hour * 60 + t.tm_min
    if start <= end:
        return start <= mins < end
    # Overnight wrap: "23:00-08:00" is silent after 23:00 AND before 08:00.
    return mins >= start or mins < end


def clip_text(text: str) -> tuple[str, bool]:
    """Truncate at a word boundary to PROACTIVE_MAX_CHARS; (text, clipped?)."""
    limit = PROACTIVE_MAX_CHARS
    if limit <= 0 or len(text) <= limit:
        return text, False
    cut = text[:limit]
    sp = cut.rfind(" ")
    if sp > 0:
        cut = cut[:sp]
    return cut, True


def check_speak_allowed(
    text: str,
    room: str,
    now: float,
    last_at: dict | None = None,
) -> str:
    """Empty string when the guards allow the speech; otherwise the reason.

    Room existence is a data question handled by the tool (it answers with
    the live rooms list); this function is about the time guards only.
    """
    if last_at is None:
        last_at = _LAST_SPEAK_AT
    if not text:
        return "empty text"
    if in_quiet_hours(now):
        return (
            f"quiet hours ({PROACTIVE_QUIET_HOURS} UTC+"
            f"{PROACTIVE_QUIET_TZ_OFFSET_H}); the rooms stay silent at night"
        )
    if room and PROACTIVE_MIN_GAP_S > 0:
        prev = last_at.get(room)
        if prev is not None:
            since = now - prev
            if since < PROACTIVE_MIN_GAP_S:
                return (
                    f"room {room!r} spoke {since:.0f}s ago "
                    f"(min gap {PROACTIVE_MIN_GAP_S:.0f}s); "
                    f"retry in {PROACTIVE_MIN_GAP_S - since:.0f}s"
                )
    return ""


# --- the speak path ---------------------------------------------------------


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _dispatch_speak(session, text: str) -> None:
    """Fire-and-forget playback, the same shape as POST /api/camera/tts.

    The tool call must not block on a /play_audio POST that can answer
    20-90 s late (measured); failures land in the log through the
    done-callback, not in the response.
    """
    task = asyncio.create_task(session._speak(text, text))
    _SPEAK_TASKS.add(task)
    task.add_done_callback(_SPEAK_TASKS.discard)
    task.add_done_callback(
        lambda fut: logger.info(f"PROACTIVE TTS task done, exc={fut.exception()}")
    )


def _dispatch_speak_all(sessions: list, text: str) -> None:
    """CHORUS: ONE synthesis, the SAME pcm bytes to every camera at once.

    Strict simultaneity is the contract (the user's words: «строго хором
    одновременно»), so the phrase is synthesised exactly once and the very
    same bytes object is handed to every session; `asyncio.gather` starts the
    `/play_audio` POSTs within microseconds of each other. Playing each room
    through its own `_speak_pcm` keeps every per-session guard intact — the
    mic mute, the echo ring, `_tts_play_end`, the ratio log — all of which a
    hand-rolled fan-out would silently skip.

    TTS is fetched through `sessions[0]` on purpose: every session reads the
    same backend/voice globals, so a second synthesis would buy nothing and
    cost one more GPU round trip.

    VERDICT: `_speak_pcm` returns `is_question` — a property of the TEXT, not
    of the playback (the trap documented 10.10.2026). The honest per-room
    outcome is the session's own `_last_play_ok`, which `_speak_pcm` sets from
    `played` on the blocking path. The chorus therefore passes `wait=True`
    (each room's POST blocks its own coroutine; the rooms still run together
    under `gather` — a chorus has no next sentence to overlap) and judges by
    `_last_play_ok`, not by the return value.
    """
    async def _chorus() -> None:
        pcm = await sessions[0]._tts_fetch(text)
        if not pcm:
            # Honest everywhere-or-nowhere: no room hears a half broadcast.
            logger.warning(
                "PROACTIVE chorus: TTS produced nothing — not spoken anywhere"
            )
            return
        results = await asyncio.gather(
            *[s._speak_pcm(pcm, text, wait=True) for s in sessions],
            return_exceptions=True,
        )
        ok: list[str] = []
        bad: list[tuple[str, str]] = []
        for s, r in zip(sessions, results):
            # `r is not True` would put EVERY successful room here: `_speak_pcm`
            # returns `is_question` — a property of the TEXT — and a phrase that
            # is not a question returns False. The chorus verdict said
            # "failed: [('livingroom', 'False'), ...]" while all five
            # `play_audio` lines showed the audio played (17.11.2026, probe #2):
            # the same is_question trap as the 10.10.2026 return-value verdict,
            # one level up. Under return_exceptions=True an exception comes back
            # as an exception INSTANCE, never as a bool — so test for that.
            if isinstance(r, BaseException):       # an exception, captured
                bad.append((s.stream_name, repr(r)))
            elif getattr(s, "_last_play_ok", False):
                ok.append(s.stream_name)
            else:
                bad.append((s.stream_name, "False"))
        msg = f"PROACTIVE chorus: spoken in {ok}"
        if bad:
            msg += f"; failed: {bad}"
        logger.info(msg)

    task = asyncio.create_task(_chorus())
    _SPEAK_TASKS.add(task)
    task.add_done_callback(_SPEAK_TASKS.discard)
    task.add_done_callback(
        lambda fut: logger.info(f"PROACTIVE chorus task done, exc={fut.exception()}")
    )


async def _speak_tool(text: str, room: str = "", now: float | None = None) -> str:
    """Body of the `speak` tool. Returns a JSON string for the MCP result."""
    if now is None:
        now = time.time()
    text = (text or "").strip()
    room = (room or "").strip()
    rooms_list = _room_names()
    if not text:
        return _json({"error": "empty text"})
    if not room:
        return _json({"error": "name a room; call rooms() first",
                      "rooms": rooms_list})
    target = [s for s in _SESSIONS if s.stream_name == room]
    if not target and room != "all":
        return _json({"error": f"unknown room {room!r}", "rooms": rooms_list})
    clipped, was_clipped = clip_text(text)
    if room == "all":
        # Strict chorus: ALL live rooms at the same moment, from one
        # synthesis. A room inside its own min-gap would break the chorus,
        # so the whole call is refused and the blocked rooms are named —
        # partial delivery («камера промолчала») is the failure this
        # all-or-nothing rule exists to prevent.
        blocked: dict = {}
        for s in _SESSIONS:
            r = check_speak_allowed(clipped, s.stream_name, now)
            if r:
                blocked[s.stream_name] = r
        if not _SESSIONS:
            return _json({"error": "no camera sessions are running",
                          "rooms": rooms_list})
        if blocked:
            return _json({"error":
                          "chorus needs every room free; these are not",
                          "blocked": blocked, "rooms": rooms_list})
        names = [s.stream_name for s in _SESSIONS]
        # The audit line: one place to answer "what did Hermes say, when, where".
        logger.info(f"🗣 PROACTIVE (hermes) [all: {','.join(names)}]: {clipped!r}")
        for name in names:
            _LAST_SPEAK_AT[name] = now
        _dispatch_speak_all(list(_SESSIONS), clipped)
        res = {"spoken": True, "room": "all", "rooms": names,
               "chars": len(clipped)}
        if was_clipped:
            res["note"] = f"text was truncated to {len(clipped)} chars"
        return _json(res)
    reason = check_speak_allowed(clipped, room, now)
    if reason:
        return _json({"error": reason, "rooms": rooms_list})
    session = target[0]
    # The audit line: one place to answer "what did Hermes say, when, where".
    logger.info(f"🗣 PROACTIVE (hermes) [{room}]: {clipped!r}")
    _LAST_SPEAK_AT[room] = now
    _dispatch_speak(session, clipped)
    res: dict = {"spoken": True, "room": room, "chars": len(clipped)}
    if was_clipped:
        res["note"] = f"text was truncated to {len(clipped)} chars"
    return _json(res)


async def stop_speak_tasks() -> None:
    """Cancel the outstanding proactive playback tasks (gateway shutdown)."""
    tasks = [t for t in list(_SPEAK_TASKS) if not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# --- server -----------------------------------------------------------------


def build_mcp_server():
    """Build the FastMCP server (v1 SDK API — `mcp<2`, see module docstring).

    Import happens here, not at module level, so the guards stay testable in
    an image that does not carry the `mcp` package (the gateway test image
    pattern: a module nobody can import is a module nobody tests).
    """
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    # host/port MUST match the real bind, and transport_security must name the
    # LAN host explicitly. With the FastMCP DEFAULTS (host=127.0.0.1) the SDK
    # auto-enables DNS-rebinding protection allowlisting ONLY localhost Host
    # headers — every real client (hermes -> 192.168.22.102:18793) got
    # `421 Misdirected Request / Invalid host header` and could never connect
    # (measured 17.11.2026: 401 fail-closed worked, tokened SSE did not).
    # Passing an explicit TransportSecuritySettings keeps the rebinding
    # protection (Browsers only; the token gates everything anyway) while
    # admitting the LAN address hermes actually connects to.
    mcp = FastMCP(
        "voice_gateway",
        host="0.0.0.0",
        port=PROACTIVE_MCP_PORT,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["192.168.22.102:*", "127.0.0.1:*", "localhost:*", "[::1]:*"],
        ),
    )

    @mcp.tool()
    async def rooms() -> str:
        """List the camera rooms that can be spoken to.

        Call this before speak(): the room name in speak() must be one of
        these, and an unknown room is refused.
        """
        return _json({"rooms": _room_names()})

    @mcp.tool()
    async def speak(text: str, room: str = "") -> str:
        """Speak one short phrase through a room's camera speaker.

        room must be one of rooms(), or "all" to speak STRICTLY IN CHORUS
        through every live camera at once (one synthesis, the same audio
        started in every room together; "all" is refused unless EVERY room
        is free). Guards: quiet hours 23:00-08:00 MSK (the rooms stay silent
        at night), at most one proactive phrase per room every 120 s, text
        over 240 chars is truncated. For announcements on our own
        initiative; replies to the user keep the normal dialogue path.
        """
        return await _speak_tool(text, room)

    return mcp


class BearerAuth:
    """Pure-ASGI Bearer-token wrapper, fail-closed on an empty token.

    Pure ASGI on purpose: it wraps FastMCP's Starlette app without depending
    on Starlette middleware internals, and a non-http scope (lifespan) passes
    through untouched so the server's own lifespan still runs.
    """

    def __init__(self, app, token: str = ""):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        provided = headers.get("authorization", "")
        scheme, _, value = provided.partition(" ")
        expected = f"Bearer {self.token}" if self.token else ""
        # Scheme is case-insensitive per RFC 7235; the token is compared in
        # constant time so a timing probe is not a probe.
        ok = (
            self.token
            and scheme.lower() == "bearer"
            and hmac.compare_digest(value.strip(), self.token)
        )
        if not ok:
            body = b'{"error": "unauthorized"}'
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def _mcp_disabled_reason() -> str:
    """Why the MCP server must not start; empty string when it may."""
    if not MCP_TOKEN:
        return "MCP_TOKEN is empty (fail-closed)"
    if PROACTIVE_MCP_PORT <= 0:
        return "PROACTIVE_MCP_PORT is 0"
    return ""


async def run_mcp_server() -> None:
    """Serve the MCP SSE server on PROACTIVE_MCP_PORT in the CURRENT loop.

    The camera sessions live on the main loop (their aiohttp session and
    asyncio state are loop-bound), so the tools must run on that same loop —
    which is why this is an asyncio task and not a thread with its own
    uvicorn.run.
    """
    why = _mcp_disabled_reason()
    if why:
        logger.info(f"🔌 Hermes MCP disabled ({why})")
        return
    try:
        mcp = build_mcp_server()
    except Exception as e:
        # Fail LOUD, not silent: a missing `mcp` package must not look like
        # a server that is simply quiet.
        logger.error(f"🔌 Hermes MCP failed to build (is 'mcp<2' installed?): {e!r}")
        return

    import uvicorn

    config = uvicorn.Config(
        BearerAuth(mcp.sse_app(), MCP_TOKEN),
        host="0.0.0.0",
        port=PROACTIVE_MCP_PORT,
        log_level="warning",
        access_log=False,
        # Hermes holds an SSE stream open; without a bound the shutdown below
        # would wait for it forever and delay the gateway's own teardown.
        timeout_graceful_shutdown=3,
    )
    server = uvicorn.Server(config)
    logger.info(
        f"🔌 Hermes MCP starting: http://0.0.0.0:{PROACTIVE_MCP_PORT}/sse "
        f"(tools: speak, rooms; guards: quiet={PROACTIVE_QUIET_HOURS} "
        f"UTC+{PROACTIVE_QUIET_TZ_OFFSET_H}, gap={PROACTIVE_MIN_GAP_S:.0f}s, "
        f"max={PROACTIVE_MAX_CHARS} chars)"
    )
    # NOT server.serve(): capture_signals() installs its own SIGTERM/SIGINT
    # handlers over the main uvicorn.run's, and the container's SIGTERM (the
    # gateway's own graceful shutdown) would be swallowed until SIGKILL.
    # Instead: uvicorn's _serve() verbatim minus that capture. The line that
    # matters is the LIFESPAN one — self.lifespan is created by serve(),
    # never by startup(), so a hand-driven server without it dies at
    # uvicorn's `await self.lifespan.startup()` with AttributeError while the
    # log shows nothing but the starting line (measured on this exact build).
    try:
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        if not server.should_exit:
            await server.main_loop()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Fail LOUD: a server task that dies silently looks exactly like a
        # healthy one that is merely quiet (the audioop NameError lesson).
        logger.error(f"🔌 Hermes MCP server failed: {e!r}")
        return
    finally:
        if server.started:
            try:
                await server.shutdown()
            except Exception as e:
                logger.error(f"🔌 Hermes MCP shutdown error: {e!r}")
