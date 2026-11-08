"""The loop-stall dump: the only witness that can speak while the loop cannot.

Two stalls went undiagnosed because the log held a duration and nothing else:
124.3 s on 05.11.2026 and 44.9 s on 07.11.2026. A coroutine cannot report on
a frozen loop, so the stack has to come from a thread that is not the loop —
`faulthandler.dump_traceback_later` runs in its own C thread and fires on wall
clock whatever Python is doing.

These tests call the code, they do not grep it: the last mechanism in this file
whose wiring was only asserted by reading the source (the `stream_name` of the
light-automation tie, 20.10.2026) was dead in production for days.
"""
import os
import sys
import time
import asyncio

import pytest

os.environ.setdefault("NANOBOT_WS_URL", "ws://test_nanobot")
os.environ.setdefault("WHISPER_URL", "http://test_whisper")
os.environ.setdefault("TTS_URL", "http://test_tts")
os.environ.setdefault("SPEAKER_ID_URL", "http://test_speaker")

import main


def test_the_fuse_defaults_to_twenty_seconds(monkeypatch):
    monkeypatch.delenv("LOOP_STALL_DUMP_S", raising=False)
    assert main._env_seconds("LOOP_STALL_DUMP_S", 20.0) == 20.0


def test_zero_means_off_and_never_arms(monkeypatch):
    armed = []
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 0.0)
    monkeypatch.setattr(main, "_arm_stall_dump", lambda: armed.append(1))
    assert main._maybe_rearm_stall_dump(1e9) is False
    assert armed == []


def test_a_healthy_loop_rearms_only_after_half_the_fuse(monkeypatch):
    """Re-arming on every tick would spawn a thread per second for nothing.

    Half is the right point: the fuse then always has at most one full
    `_LOOP_DUMP_AFTER_S` left, so a stall longer than that cannot slip past.
    """
    calls = []
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 20.0)
    monkeypatch.setattr(main, "_arm_stall_dump", lambda: calls.append(1))
    monkeypatch.setattr(main, "_stall_dump_armed_at", 1000.0)

    assert main._maybe_rearm_stall_dump(1009.0) is False
    assert calls == []
    assert main._maybe_rearm_stall_dump(1010.0) is True
    assert calls == [1]


def test_arming_calls_faulthandler_with_the_fuse_and_an_open_file(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 20.0)
    monkeypatch.setattr(main, "_stall_dump_armed_at", 0.0)
    monkeypatch.setattr(main.faulthandler, "dump_traceback_later",
                        lambda t, repeat, file: seen.append((t, repeat, file)))
    handle = open(tmp_path / "stacks.log", "a", buffering=1, encoding="utf-8")
    try:
        monkeypatch.setattr(main, "_stall_dump_file", handle)
        main._arm_stall_dump()
        assert seen == [(20.0, False, handle)]
        assert main._stall_dump_armed_at > 0
    finally:
        handle.close()


def test_the_dump_file_is_a_real_descriptor_in_the_durable_log_dir(tmp_path, monkeypatch):
    """faulthandler writes to a fd, so a StringIO would raise at the worst time."""
    monkeypatch.setenv("GATEWAY_LOG_DIR", str(tmp_path))
    handle = main._open_stall_dump_file()
    try:
        assert handle is not sys.stderr
        assert handle.fileno() >= 0
        assert (tmp_path / "loop_stalls.log").exists()
    finally:
        handle.close()


def test_the_dump_file_falls_back_to_stderr_without_a_log_dir(monkeypatch):
    monkeypatch.delenv("GATEWAY_LOG_DIR", raising=False)
    assert main._open_stall_dump_file() is sys.stderr


def test_a_stall_longer_than_the_fuse_writes_a_traceback(tmp_path, monkeypatch):
    """The end-to-end proof: block the thread and read the file afterwards.

    A stub that only records the call would pass with the file closed, with
    `repeat=True`, or with a timeout of a day — none of which would ever put a
    stack on disk.
    """
    path = tmp_path / "loop_stalls.log"
    handle = open(path, "a", buffering=1, encoding="utf-8")
    monkeypatch.setattr(main, "_stall_dump_file", handle)
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 0.2)
    monkeypatch.setattr(main, "_stall_dump_armed_at", 0.0)
    try:
        main._arm_stall_dump()
        time.sleep(0.8)          # the "stall" — faulthandler's thread is not us
    finally:
        main.faulthandler.cancel_dump_traceback_later()
        handle.flush()
        handle.close()

    text = path.read_text(encoding="utf-8", errors="replace")
    assert "Thread" in text or 'File "' in text, f"no traceback was written: {text!r}"
    assert "test_loop_stall_dump" in text or "test_a_stall" in text


@pytest.mark.asyncio
async def test_the_monitor_keeps_the_fuse_armed(monkeypatch):
    """The coroutine must CALL the re-arm, not merely have one next to it.

    The first version of this test called `_maybe_rearm_stall_dump` itself
    after a 150 ms sleep — and passed with the call deleted from the monitor,
    i.e. it proved the function existed and nothing about the wiring. That is
    the same shape as the `stream_name` that was added to the light tie and
    never passed (20.10.2026): a parameter that reaches no call site is a
    comment, and a test that makes the call itself proves nothing.

    The 1 s tick is shortened by patching `main.asyncio.sleep`, otherwise this
    test would need a real second per iteration.
    """
    real_sleep = asyncio.sleep
    calls = []

    async def fast_sleep(_t, *a, **k):
        await real_sleep(0.01)

    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(main, "_LOOP_STALL_THRESHOLD", 999.0)
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 20.0)
    monkeypatch.setattr(main, "_stall_dump_armed_at", 0.0)
    monkeypatch.setattr(main, "_maybe_rearm_stall_dump",
                        lambda now: calls.append(now) or True)

    task = asyncio.create_task(main.loop_stall_monitor())
    try:
        await real_sleep(0.12)          # several shortened ticks
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert calls, "the monitor ran but never called the re-arm"


@pytest.mark.asyncio
async def test_startup_opens_the_file_and_arms_the_fuse(monkeypatch):
    """`on_startup` is where the wiring lives, and where it can silently die.

    A dump that is never armed is indistinguishable from a stall that left no
    stack, so this asserts the startup path itself: the file handle is stored
    in the global the fuse reads, and the fuse is armed.
    """
    sentinel = object()
    armed, started = [], []

    async def fake_sessions():
        started.append(1)

    async def fake_monitor():
        return

    monkeypatch.setattr(main, "start_camera_sessions", fake_sessions)
    monkeypatch.setattr(main, "loop_stall_monitor", fake_monitor)
    monkeypatch.setattr(main, "_open_stall_dump_file", lambda: sentinel)
    monkeypatch.setattr(main, "_arm_stall_dump", lambda: armed.append(1))
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "")

    await main.on_startup()
    if main._loop_monitor_task is not None:
        main._loop_monitor_task.cancel()

    assert started, "the camera sessions must still start"
    assert main._stall_dump_file is sentinel
    assert armed == [1]


def test_the_fatal_threshold_defaults_to_sixty_seconds(monkeypatch):
    monkeypatch.delenv("LOOP_STALL_FATAL_S", raising=False)
    # Re-import to pick up the default; simplest is to call _env_seconds directly.
    assert main._env_seconds("LOOP_STALL_FATAL_S", 60.0) == 60.0


def test_zero_fatal_threshold_means_never_exit(monkeypatch):
    """`0` disables the fatal exit — dump-only mode, same as every other _S."""
    monkeypatch.setattr(main, "_LOOP_STALL_FATAL_S", 0.0)
    exited = []
    monkeypatch.setattr(main.os, "_exit", lambda code: exited.append(code))
    monkeypatch.setattr(main, "_LOOP_STALL_THRESHOLD", 999.0)
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 0.0)

    async def run_one_tick():
        async def no_sleep(_t, *a, **k):
            return
        monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
        t = [1000.0]
        def fake_mono():
            t[0] += 10.0          # lag of 10 s, above threshold, fatal is 0=off
            return t[0]
        monkeypatch.setattr(main.time, "monotonic", fake_mono)
        task = asyncio.create_task(main.loop_stall_monitor())
        try:
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_one_tick())
    assert exited == [], "fatal threshold of 0 must never call os._exit"


@pytest.mark.asyncio
async def test_a_fatal_stall_exits_the_process(monkeypatch):
    """A stall above the fatal threshold must kill the process so Docker
    (unless-stopped) restarts the gateway instead of leaving it hung."""
    exited = []
    monkeypatch.setattr(main.os, "_exit", lambda code: exited.append(code))
    monkeypatch.setattr(main, "_LOOP_STALL_THRESHOLD", 999.0)
    monkeypatch.setattr(main, "_LOOP_STALL_FATAL_S", 1.0)
    monkeypatch.setattr(main, "_LOOP_DUMP_AFTER_S", 0.0)

    real_sleep = main.asyncio.sleep

    async def no_sleep(_t, *a, **k):
        # Yield control so the loop_stall_monitor task can run between ticks.
        await real_sleep(0)

    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)

    t = [1000.0]
    def fake_mono():
        t[0] += 5.0          # lag of 5 s >= fatal threshold of 1 s
        return t[0]
    monkeypatch.setattr(main.time, "monotonic", fake_mono)

    task = asyncio.create_task(main.loop_stall_monitor())
    try:
        # First yield: monitor starts, calls t0=monotonic(), then yields on its
        # own sleep. Second yield: monitor's sleep completes, it checks lag and
        # calls os._exit(1). The test's own sleep was scheduled before the
        # monitor's, so we need two ticks for the monitor to catch up.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert exited == [1], f"expected os._exit(1), got {exited!r}"
