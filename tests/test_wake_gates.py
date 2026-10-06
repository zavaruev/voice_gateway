"""End-to-end tests for the wake-gate cascade inside `CameraSession._vad_process`.

Two of the cascade's STT-confirm rescue bands were unreachable after the
threshold recalibration: the appliance hold tested `top >= 0.85` behind a
`< 0.60` guard, the quiet-source hold tested `>= 0.60` behind a `< 0.55/0.58`
guard (and re-read the score ring *after* clearing it). The fix moved both
bands to sit just under their guards — `[0.55, 0.60)` for the appliance hold
and `[tier-0.05, tier)` for the quiet-source hold. These tests pin the bands,
the pass-through above each guard, and the idempotence of
`_open_stt_confirm()` when two gates hit their bands on the same chunk.

The engine, the utterance VAD and the arbiter are stubbed, so no model files
and no network are needed.

NOTE: imports `av` (through camera_client) — like `tests/test_camera_client.py`
this module only runs inside the Docker image (see AGENTS/README).
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import camera_client
from camera_client import CameraConfig, CameraSession


class _SilentVAD:
    """Always reports "not speech".

    The wake path uses the engine's own VAD gate, so keeping utterance
    collection off keeps `_process_utterance` (Whisper) out of these tests.
    """

    async def is_speech(self, _pcm: bytes):
        return False, None


def _make_session(bg_median: float = 0.0) -> CameraSession:
    """A session whose audio pipeline is wired for a deterministic gate run."""
    s = CameraSession(
        CameraConfig(
            stream_name="test_stream",
            go2rtc_host="127.0.0.1",
            go2rtc_port=1984,
        )
    )
    s._vad = _SilentVAD()
    # Non-None model + past warm-up/suppression is what opens the oww block.
    s._engine.oww_model = object()
    s._engine._ww_vad_speech = lambda *_a, **_k: True
    s._engine.check_wakeword = lambda *_a, **_k: None
    s._audio_epoch = time.time() - 60.0
    s._wake_suppress_until = 0.0
    s._out_track = None
    s._speaking_until = 0.0
    s._bg_window = [int(bg_median)] * 60 if bg_median else []
    # Isolate from whatever other tests left behind in the module globals.
    camera_client._ROOM_PEAKS.clear()
    camera_client._ARB_STATE["claims"].clear()
    camera_client._ARB_STATE["sent"].clear()
    camera_client._ARB_STATE["owner"] = None
    camera_client._ARB_STATE["cmd_sent"] = None
    return s


def _chunk(peak: int) -> bytes:
    """100 ms spike train with exactly `peak` as its raw maximum.

    The peak must land in [600, 4000): below 600 the engine never scores the
    chunk at all, at/above 4000 the dense-impact bang gate zeroes the score,
    and at/above 3000 `_proximity_level()` leaves the `< 3000` gates.
    """
    a = np.zeros(1600, dtype=np.int16)
    a[::10] = peak
    return a.tobytes()


@contextlib.contextmanager
def _gates_patched(session: CameraSession):
    """Stub everything the cascade touches besides the gates themselves."""
    with (
        patch(
            "camera_client._arbiter_submit",
            new=AsyncMock(return_value=(True, (), "test")),
        ),
        patch.object(session, "_play_attention", new=AsyncMock()),
        patch.object(session, "_schedule_wake_greeting"),
        patch.object(session, "_confirm_expiry_watch", new=AsyncMock()) as expiry,
    ):
        yield expiry


async def _feed(session: CameraSession, score: float, peak: int, chunks: int = 2):
    """Run `chunks` live-feed chunks through the pipeline at one oww score.

    Two chunks are the documented debounce: the first only arms `_ww_consec`,
    the second passes the `>= 2` fire check and reaches the gate cascade.
    """
    for _ in range(chunks):
        session._engine.last_score = score
        await session._vad_process(_chunk(peak))
    # Let tasks created by the cascade (attention pip, expiry watcher) run.
    await asyncio.sleep(0)


def _confirm_open(session: CameraSession) -> bool:
    return session._stt_confirm_until > time.time()


# --- appliance hold: bg median > 800 -------------------------------------

@pytest.mark.asyncio
async def test_appliance_hold_rescues_score_just_below_bar():
    s = _make_session(bg_median=1200)
    with _gates_patched(s):
        # Peak 3500 keeps the session out of the <3000 gates, so only the
        # appliance hold can act on this chunk.
        await _feed(s, score=0.57, peak=3500)
        assert _confirm_open(s), "0.57 under appliance noise must reach STT"
        assert s._wake_detected is False
        assert s._ww_consec == 0


@pytest.mark.asyncio
async def test_appliance_hold_drops_score_below_rescue_band():
    s = _make_session(bg_median=1200)
    with _gates_patched(s):
        await _feed(s, score=0.50, peak=3500)
        assert not _confirm_open(s), "0.50 is too weak to warrant a Whisper call"
        assert s._wake_detected is False


@pytest.mark.asyncio
async def test_appliance_gate_fires_at_or_above_bar():
    s = _make_session(bg_median=1200)
    with _gates_patched(s):
        await _feed(s, score=0.62, peak=3500)
        assert s._wake_detected is True
        assert not _confirm_open(s)


# --- quiet-source hold: my_lvl < 3000 ------------------------------------

@pytest.mark.asyncio
async def test_quiet_source_hold_rescues_near_tier_score():
    """Peak 1500 -> level < 2000 -> tier 0.58, band [0.53, 0.58)."""
    s = _make_session()
    with _gates_patched(s):
        await _feed(s, score=0.56, peak=1500)
        assert _confirm_open(s), "0.56 vs tier 0.58 must reach STT"
        assert s._wake_detected is False


@pytest.mark.asyncio
async def test_quiet_source_far_tier_rescues_near_tier_score():
    """Peak 2500 -> level in [2000, 3000) -> tier 0.55, band [0.50, 0.55)."""
    s = _make_session()
    with _gates_patched(s):
        await _feed(s, score=0.52, peak=2500)
        assert _confirm_open(s), "0.52 vs tier 0.55 must reach STT"
        assert s._wake_detected is False


@pytest.mark.asyncio
async def test_quiet_source_hold_drops_score_below_rescue_band():
    s = _make_session()
    with _gates_patched(s):
        await _feed(s, score=0.50, peak=1500)
        assert not _confirm_open(s), "below the band the copy is not the user"
        assert s._wake_detected is False


@pytest.mark.asyncio
async def test_quiet_source_hold_passes_score_at_tier():
    """A score that clears the tier must fire, not be held for STT."""
    s = _make_session()
    with _gates_patched(s):
        await _feed(s, score=0.58, peak=1500)
        assert s._wake_detected is True
        assert not _confirm_open(s)


# --- _open_stt_confirm ----------------------------------------------------

@pytest.mark.asyncio
async def test_two_gates_on_one_chunk_open_a_single_confirm_window():
    """Appliance hold runs first and clears the score ring, which lets the
    quiet-source gate re-read the bare chunk score inside its own band too.
    Both must share one window — a second expiry watcher would greet the same
    silent room twice."""
    s = _make_session(bg_median=1200)
    with _gates_patched(s) as expiry:
        await _feed(s, score=0.56, peak=1500)
        assert _confirm_open(s)
        assert expiry.call_count == 1
        assert s._wake_detected is False


@pytest.mark.asyncio
async def test_open_stt_confirm_does_not_extend_an_open_window():
    s = _make_session()
    s._stt_confirm_until = time.time() + 5.0
    with _gates_patched(s) as expiry:
        s._open_stt_confirm(0.9)
        assert expiry.call_count == 0
        assert s._stt_confirm_until <= time.time() + 5.0


@pytest.mark.asyncio
async def test_open_stt_confirm_opens_a_fresh_window():
    s = _make_session()
    assert s._stt_confirm_until == 0.0
    with _gates_patched(s) as expiry:
        s._open_stt_confirm(0.57)
        assert _confirm_open(s)
        assert expiry.call_count == 1
        await asyncio.sleep(0)


# --- the wake gate cascade's sibling: ENERGY, not keywords --------------------
#
# Measured 06.10.2026 08:55 on the kitchen, whose microphone is dead
# (rms 4, -77.8 dB, peak 135-388 over ten seconds):
#
#     VOSK WAKE trig='компьютер'
#     Whisper IN: 1.15s raw_rms=77  peak=388  dB=-52.6  gain=20.0x
#     Whisper OK: 'Выключи.'
#
# A decoder run on near-silence returns its most likely phrase, and for this
# system that phrase is «компьютер». The room woke itself, Whisper invented a
# command out of the same silence, and the camera spoke the invention — which is
# the whole of the "answered unintelligibly and did nothing" report.


class _StubVosk:
    """A vosk matcher that always decodes the wake word.

    Carries the three counters `vosk diag` reads, because that log line runs on
    the same chunk as the decision under test and an incomplete stub fails on the
    logging rather than on the gate.
    """

    active = True
    triggers = 0
    decodes = 0
    last_partial = ""
    last_text = "компьютер"
    last_trigger_text = "компьютер"

    def __init__(self):
        self.fed = 0
        self.resets = 0

    def begin(self):
        self.active = True

    def feed(self, chunk):
        self.fed += 1
        self.triggers += 1
        return True

    def reset(self):
        self.resets += 1


async def _feed_quiet(session: CameraSession, peak: int, rounds: int = 1):
    with (
        patch("camera_client._arbiter_submit", new=AsyncMock(return_value=(True, (), "t"))),
        patch.object(session, "_play_attention", new=AsyncMock()),
        patch.object(session, "_schedule_wake_greeting"),
        patch.object(session, "_fire_wake", new=AsyncMock()) as fire,
    ):
        for _ in range(rounds):
            await session._vad_process(_chunk(peak))
        await asyncio.sleep(0)
    return fire


@pytest.mark.asyncio
async def test_a_wake_decoded_from_silence_does_not_fire():
    """A dead microphone cannot contain a word. The kitchen measured a peak of
    388 while its decoder reported «компьютер» with full confidence."""
    s = _make_session()
    s._vosk_wake = _StubVosk()

    fire = await _feed_quiet(s, peak=388)

    assert fire.await_count == 0, (
        "a wake word decoded out of near-silence fired the room"
    )
    assert s._vosk_wake.resets >= 1, "the decoder must be rolled over, not left armed"


@pytest.mark.asyncio
async def test_the_energy_gate_does_not_swallow_a_real_command():
    """The user's own commands measure 8000-32767 at the rail in the living room,
    so the threshold must sit far below them — this is the failure a fix here
    would cause, not just the one it prevents."""
    s = _make_session()
    s._vosk_wake = _StubVosk()

    fire = await _feed_quiet(s, peak=12000)

    assert fire.await_count == 1, "a genuine command peak was refused"


@pytest.mark.asyncio
async def test_the_energy_gate_is_per_room_and_off_by_default_for_zero():
    """`0` must mean "use the built-in default", never "refuse everything" — a
    half-filled override that zeroed the threshold would silence the whole house
    and read as a dead wake word."""
    assert CameraConfig(stream_name="x").wake_min_peak == 0
    s = _make_session()
    assert s._wake_min_peak == camera_client._WAKE_MIN_PEAK > 0

    s2 = CameraSession(
        CameraConfig(stream_name="y", go2rtc_host="127.0.0.1", wake_min_peak=800)
    )
    assert s2._wake_min_peak == 800
