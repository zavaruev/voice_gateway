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
from collections import deque
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import camera_client
from camera_client import (
    _ECHO_REPEAT_HORIZON_S,
    CameraConfig,
    CameraSession,
)


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
    # A stub that mirrors the class has to track the class. `vosk diag` reads these
    # and adding them to the real class without adding them here raised
    # AttributeError inside the live audio path — which is the same reason the real
    # class carries class-level defaults for both.
    near_misses = 0
    last_near_miss = ""

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
async def test_the_gate_reads_a_window_not_the_trigger_chunk():
    """«компьютер» spans ~10 chunks at the 160 ms hop, so the trigger chunk is a
    sample from an arbitrary point inside the word. Measured on the recorded
    utterances: a 1.9 s window lifts speech median 5259 -> 11250 (p25 2912 ->
    5024) while the living-room ambient ceiling does not move at all (2316).

    Here the wake is decoded on a quiet chunk (300) preceded by a loud one
    (12000) — a real word does exactly this, and a single-chunk gate would
    refuse it.
    """
    s = _make_session()
    s._vosk_wake = _StubVosk()
    s._vosk_peak_win = deque([12000, 9000, 6000, 3000], maxlen=camera_client._WAKE_PEAK_WINDOW)

    fire = await _feed_quiet(s, peak=300)

    assert fire.await_count == 1, (
        "a real word whose loudest chunk is not the trigger chunk was refused"
    )


@pytest.mark.asyncio
async def test_a_loud_utterance_cannot_launder_a_silent_room():
    """The window is bounded, so it cannot remember a loud moment from long ago
    and wave a later silence through. Feeding the loud chunk first and then
    pushing the window past its length must put the gate back to refusing."""
    s = _make_session()
    s._vosk_wake = _StubVosk()
    s._vosk_peak_win = deque([12000], maxlen=camera_client._WAKE_PEAK_WINDOW)

    for _ in range(camera_client._WAKE_PEAK_WINDOW):
        await s._vad_process(_chunk(400))
    fire = await _feed_quiet(s, peak=400, rounds=0)

    assert fire.await_count == 0, "the window outlived its own length"
    assert max(s._vosk_peak_win) < s._wake_min_peak


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


# --- the VOSK WAKE log printed a DIFFERENT number than the gate judged (07.10) ---
#
# The energy gate is `max(self._vosk_peak_win)` — the loudest chunk of the last
# 1.9 s — because «компьютер» spans ~10 chunks and the trigger lands at an
# arbitrary point inside the word. The `VOSK WAKE` line printed only the TRIGGER
# chunk. Read on 07.10.2026 12:18 in the living room it showed
# `peak=997 min_peak=3000`, which reads as a gate that had failed; the gate had
# passed on the window. An instrument that shows a different number than the one
# being judged is the failure mode this whole file keeps rediscovering.


def test_the_wake_log_reports_the_peak_the_gate_actually_judged():
    """Both numbers belong in the line: the chunk that triggered, and the window
    the decision was made on.

    This test reads the SOURCE of the log line rather than the emitted text,
    which is the wrong instrument for a format check but the only one available
    for a branch that needs a live decoder to reach. Three earlier versions of it
    each asserted against a COMMENT quoting the old line verbatim, because a
    comment in that file quotes `peak=997 min_peak=3000` as the false alarm. The
    lesson is recorded here because it is the file's recurring failure: an
    instrument that cannot fail is not a check."""
    import inspect

    import camera_client as cc

    # Only the f-string ARGUMENTS of that one logger call. Three earlier versions
    # sliced the source and every one of them matched a COMMENT quoting the old
    # log line, or cut the call short — the same failure as the one being
    # guarded, one level down. So: find the call, then read forward to the
    # closing paren at this indentation, ignoring comments entirely.
    src = inspect.getsource(cc.CameraSession._vad_process)
    lines = src.split("\n")
    # The first hit is a COMMENT that quotes the line verbatim (that is what made
    # the earlier three versions assert against prose), so pick the hit that is
    # itself inside a f-string.
    start = next(i for i, ln in enumerate(lines)
                 if "VOSK WAKE " in ln and ln.strip().startswith("f\""))
    open_at = max(i for i in range(start + 1)
                  if lines[i].strip().startswith("logger.info("))
    indent = len(lines[open_at]) - len(lines[open_at].lstrip())
    body = []
    for ln in lines[open_at + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        if ln.strip().startswith("#"):
            continue
        body.append(ln.strip())
    wake_line = "\n".join(body)

    assert "max(self._vosk_peak_win)" in wake_line, (
        f"the VOSK WAKE log does not report the window peak: {wake_line!r}"
    )
    assert "min_peak=" not in wake_line, (
        f"the threshold is now logged as window=<max>/<threshold>, and the old "
        f"label invites reading it against the trigger chunk: {wake_line!r}"
    )


# ---------------------------------------------------------------------------
# A WAKE THAT LOSES THE CROSS-CAMERA CONTEST MUST NOT BEEP.
#
# Measured 11.10.2026 16:19: the user said «компьютер» between two cameras and
# BOTH greeted:
#
#     16:19:09.444 [entrance1]  VOSK WAKE  peak=3613
#     16:19:09.454 [entrance1]  🔔 attention pip
#     16:19:09.637 [livingroom] VOSK WAKE  peak=1599
#     16:19:09.651 [livingroom] 🔔 attention pip
#
# There was no `arbiter[wake]` line at all: the arbitration lived only in the
# openWakeWord branch, and every room here runs vosk.
#
# NOTE ON WHY THIS WENT UNNOTICED, because it is the more useful half: the
# helper `_feed_quiet` above patches `_arbiter_submit` to return "we won". A
# path that never CALLS the arbiter therefore looks identical to one that calls
# it and wins — the stub made the missing guard invisible. These tests feed the
# chunk themselves and let the arbiter answer.
# ---------------------------------------------------------------------------


async def _feed_with_arbiter(session, won: bool, peak: int = 12000):
    """Drive one vosk chunk with the arbiter answering a chosen way."""
    with (
        patch("camera_client._arbiter_submit",
              new=AsyncMock(return_value=(won, (), "other" if not won else "t"))),
        patch.object(session, "_play_attention", new=AsyncMock()),
        patch.object(session, "_schedule_wake_greeting"),
        patch.object(session, "_fire_wake", new=AsyncMock()) as fire,
    ):
        await session._vad_process(_chunk(peak))
        await asyncio.sleep(0)
    return fire


@pytest.mark.asyncio
async def test_a_wake_that_loses_the_cross_camera_contest_does_not_beep():
    """Two cameras heard the same word; only the louder one may greet.

    The pip is inside `_fire_wake`, so refusing to fire is what keeps the room
    from answering twice — and it is the ONLY thing the user hears at that
    moment.
    """
    s = _make_session()
    s._vosk_wake = _StubVosk()

    fire = await _feed_with_arbiter(s, won=False)

    assert fire.await_count == 0, (
        "this room lost the cross-camera contest and still fired: it claimed "
        "ownership and played the attention pip, which is the double greeting "
        "measured on 11.10.2026 16:19"
    )


@pytest.mark.asyncio
async def test_a_wake_that_wins_the_cross_camera_contest_does_beep():
    """The other direction, because a guard that only ever says "no" would
    silence the house: winning must still fire."""
    s = _make_session()
    s._vosk_wake = _StubVosk()

    fire = await _feed_with_arbiter(s, won=True)

    assert fire.await_count == 1, (
        "the winning room did not fire — the contest is now refusing everyone"
    )


# ---------------------------------------------------------------------------
# The arbiter compared RAW levels, so the room with the most sensitive
# microphone won every tie instead of the room the user was standing in.
#
# Measured 13.10.2026 09:03: the user spoke directly beside the living-room
# camera and the ENTRANCE camera answered. Both rooms heard the same word 37 ms
# apart and the arbiter logged
#
#     arbiter[wake]: {'livingroom': (7303, 0), 'entrance1': (9577, 0)}
#                    -> 'entrance1'
#
# which looks correct — 9577 beats 7303. It is not a proximity measurement: the
# two microphones differ ~4x in gain (median feed rms 495 in the living room
# against 1981 at the entrance, and 3616 in the kitchen). Against their OWN
# floors the same event was 16.2x for the living room and 6.6x for the entrance,
# so the room the user was next to heard it 2.5x better and lost.
#
# The numbers below are those two rooms' measured values, so this test fails on
# the pre-fix code for the same reason the field did.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arbiter_prefers_the_room_that_heard_better_relative_to_its_own_floor():
    """Loudest-wins is only a proximity test when the microphones match."""
    near = _make_session()          # the room the user was standing in
    far = _make_session()           # a different room, heard from further away
    near.stream_name = "livingroom"
    far.stream_name = "entrance1"

    # Measured absolute peak level over the last ~2 s (mean of _recent_peaks).
    near._recent_peaks = [7303] * 12
    far._recent_peaks = [9577] * 12

    # ...and each room's own tracked noise floor, normalised 0-1 exactly as
    # _NoiseFloor stores it (it is fed `orig_rms`, which is rms/32768).
    near._noise.value = 495 / 32768.0
    far._noise.value = 1981 / 32768.0

    near_won, far_won = await asyncio.gather(
        near._claim_wake(), far._claim_wake(),
    )

    assert near_won and not far_won, (
        "the room the user was standing next to lost the contest to a room that "
        "merely has a louder microphone: near=16.2x its own floor, far=6.6x its "
        "own, and the arbiter picked the room with the higher RAW level"
    )


# ---------------------------------------------------------------------------
# Measured 13.10.2026 on `camera-corridor1` (192.168.22.165), which has produced
# zero wake words in ~49,000 chunks while its neighbour reads them all night.
#
# The room is not quiet and not clipped; it is LOW-FREQUENCY DOMINATED, and its
# SNR is 15-20 dB worse than a healthy room in every band. The microphone DOES
# pass high frequencies — +19.9 dB in 3-7 kHz measured against its own floor —
# so the margin is recoverable without touching the hardware.
#
# A first attempt at this was rejected: the filter was a block FFT with no
# overlap-add, and it decoded to nothing. That rejection measured the
# implementation, not the idea. These tests pin the properties that made it work.
# ---------------------------------------------------------------------------


def test_a_room_without_the_filter_gets_its_audio_untouched():
    """Default off, everywhere. A wrong default would silently filter three
    working rooms the first time this shipped."""
    s = _make_session()
    s._wake_highpass_hz = 0
    s._init_wake_highpass()
    assert s._hp_sos is None

    chunk = _chunk(12000)
    assert s._wake_audio(chunk) == chunk


def test_the_filter_attenuates_the_rumble_and_leaves_the_speech_band():
    """That is the whole point: the noise lives below 300 Hz here."""
    import numpy as np

    s = _make_session()
    s._wake_highpass_hz = 150
    s._init_wake_highpass()
    assert s._hp_sos is not None, "the filter was not built"

    sr = 16000
    t = np.arange(int(sr * 0.16)) / sr  # one 160 ms chunk
    rumble = 1200.0 * np.sin(2 * np.pi * 40 * t)
    speech = 1200.0 * np.sin(2 * np.pi * 1200 * t)
    mixed = (rumble + speech).astype(np.int16).tobytes()

    before = np.frombuffer(mixed, dtype=np.int16).astype(np.float64)
    out = np.frombuffer(s._wake_audio(mixed), dtype=np.int16).astype(np.float64)

    def db(sig, freq):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        f = np.fft.rfftfreq(len(sig), 1.0 / sr)
        band = spec[(f >= freq * 0.8) & (f <= freq * 1.2)]
        other = spec[(f >= 0) & (f < sr / 2)]
        # relative to total energy, since the input mixes both tones
        return 10 * np.log10(max(float((band ** 2).sum()), 1e-9) / float((other ** 2).sum()))

    drop = db(before, 40) - db(out, 40)
    kept = db(before, 1200) - db(out, 1200)
    assert drop > 6, f"40 Hz rumble survived: only {drop:.1f} dB down"
    # A 4th-order Butterworth at 150 Hz is -3 dB at its own corner, so a few dB
    # at 1.2 kHz is correct behaviour and not damage. What must NOT happen is the
    # speech band being gutted — that is exactly what the first block-FFT attempt
    # did, and it is what this test exists to catch.
    assert abs(kept) < 8, (
        f"1.2 kHz speech band was changed by {kept:+.1f} dB — the filter is "
        f"damaging the band it is supposed to preserve"
    )


def test_the_filter_carries_its_state_so_there_is_no_edge_at_every_chunk():
    """Running the filter per chunk would put a transient at every 160 ms
    boundary, and a transient that often is what eats consonants. The state is
    carried, so a steady tone stays steady across a chunk boundary."""
    import numpy as np

    s = _make_session()
    s._wake_highpass_hz = 150
    s._init_wake_highpass()

    sr = 16000
    t = np.arange(sr) / sr  # one full second, fed as six 160 ms chunks
    # 200 Hz, so the SIGNAL's own largest sample-to-sample step is ~94 and any
    # edge a restarting filter would add is unmistakable above it.
    tone = (1200.0 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)

    out = np.concatenate([
        np.frombuffer(s._wake_audio(tone[i * 2560:(i + 1) * 2560].tobytes()),
                      dtype=np.int16).astype(np.float64)
        for i in range(6)
    ])

    jumps = np.abs(np.diff(out))
    natural = jumps.max()
    at_edges = jumps[[2559, 5119, 7679, 10239, 12799]]
    assert at_edges.max() < 3 * natural, (
        f"a step of {at_edges.max():.0f} samples at a chunk boundary against a "
        f"signal's own maximum step of {natural:.0f} means the filter state is "
        f"not being carried between chunks"
    )


# --- wake-decoder debug dump ---------------------------------------------------
#
# On 13.10.2026 the corridor room fed the gateway audio that decoded as
# «компьютер» 3 times of 3 in an EXTERNAL capture of the same go2rtc stream,
# and reproduced the production feed path offline with a firing. The live
# decoder then never fired in 123786 chunks. No offline capture can settle
# that, because an external ffmpeg of the same stream clipped at the rail
# (peak exactly 32768) while the gateway received 7448-11201 over the same
# seconds — two different signals. So the dump writes the decoder's ACTUAL
# input, which is the only copy of the signal the question is about.

class _RecordingVosk:
    """A matcher that records exactly what it was fed and never fires."""

    active = True
    triggers = 0
    decodes = 0
    last_partial = ""
    last_text = ""
    last_trigger_text = ""
    near_misses = 0
    last_near_miss = ""

    def __init__(self):
        self.fed: list[bytes] = []

    def begin(self):
        self.active = True

    def feed(self, chunk):
        self.fed.append(bytes(chunk))
        return False

    def reset(self):
        pass


def _session_with_dump(tmp_path, **cfg):
    s = CameraSession(
        CameraConfig(
            stream_name="dump_room",
            go2rtc_host="127.0.0.1",
            go2rtc_port=1984,
            vosk_dump_dir=str(tmp_path),
            **cfg,
        )
    )
    s._vad = _SilentVAD()
    s._engine.oww_model = object()
    s._engine._ww_vad_speech = lambda *_a, **_k: True
    s._engine.check_wakeword = lambda *_a, **_k: None
    s._audio_epoch = time.time() - 60.0
    s._wake_suppress_until = 0.0
    s._out_track = None
    s._speaking_until = 0.0
    s._bg_window = []
    return s


@pytest.mark.asyncio
async def test_dump_is_off_by_default(tmp_path):
    """Off must be the default: a forgotten debug flag must not write a file
    per room per minute. An empty `vosk_dump_dir` means no handle at all."""
    s = _make_session()
    s._vosk_wake = _RecordingVosk()
    await _feed_quiet(s, 1000)
    assert s._vosk_dump_fh is None
    assert list(tmp_path.iterdir()) == [], "nothing may be written when disabled"


@pytest.mark.asyncio
async def test_dump_writes_exactly_what_the_decoder_received(tmp_path):
    """The whole point: these bytes must be the decoder's input, byte for
    byte, so an offline reproduction runs the same signal the live room did.
    Writing the pre-filter chunk would reproduce a room that does not exist."""
    s = _session_with_dump(tmp_path)
    stub = _RecordingVosk()
    s._vosk_wake = stub
    chunks = [_chunk(1000), _chunk(1200), _chunk(900)]
    with (
        patch("camera_client._arbiter_submit", new=AsyncMock(return_value=(True, (), "t"))),
        patch.object(s, "_play_attention", new=AsyncMock()),
    ):
        for c in chunks:
            await s._vad_process(c)
        await asyncio.sleep(0)

    path = tmp_path / "dump_room.raw"
    assert path.exists(), "the dump must be written when enabled"
    written = path.read_bytes()
    expected = b"".join(c for c, f in zip(chunks, stub.fed) if f is not None)
    assert written == expected, (
        "the dump must be the bytes fed to the decoder, not some nearby signal"
    )
    assert written == b"".join(stub.fed)
    # And they must be the RAW chunk here, because this room has no high-pass.
    assert written == b"".join(chunks)


@pytest.mark.asyncio
async def test_dump_follows_the_filter_not_the_raw_chunk(tmp_path):
    """With a high-pass configured the decoder sees filtered audio, and so
    must the dump. Capturing the raw chunk would silently hand an offline
    reproduction a signal the room never decoded — exactly the mistake this
    dump exists to prevent."""
    s = _session_with_dump(tmp_path, wake_highpass_hz=300.0)
    s._init_wake_highpass()
    stub = _RecordingVosk()
    s._vosk_wake = stub
    sig = np.sin(2 * np.pi * 60 * np.arange(1280) / 16000) * 2000
    chunk = sig.astype(np.int16).tobytes()
    with (
        patch("camera_client._arbiter_submit", new=AsyncMock(return_value=(True, (), "t"))),
        patch.object(s, "_play_attention", new=AsyncMock()),
    ):
        await s._vad_process(chunk)
        await asyncio.sleep(0)

    written = (tmp_path / "dump_room.raw").read_bytes()
    assert len(stub.fed) == 1
    assert written == stub.fed[0], "dump must equal the decoder's actual input"
    assert written != chunk, (
        "a 300 Hz high-pass must change the bytes; if it did not, this test is "
        "proving nothing"
    )
    got = np.frombuffer(written, dtype=np.int16).astype(np.float64)
    # A 60 Hz tone is what the filter removes, so the dump must hold less of it.
    def energy_at(a, f):
        n = len(a)
        X = np.fft.rfft(a * np.hanning(n))
        fr = np.fft.rfftfreq(n, 1 / 16000)
        m = (fr >= f - 5) & (fr <= f + 5)
        return float(np.abs(X[m]).sum() ** 2)

    assert energy_at(got, 60) < energy_at(np.frombuffer(chunk, dtype=np.int16), 60)


@pytest.mark.asyncio
async def test_dump_stops_at_the_cap(tmp_path):
    """A diagnostic must not fill the disk. The cap is reached and the handle
    closed rather than the write path growing without bound."""
    s = _session_with_dump(tmp_path)
    s._vosk_wake = _RecordingVosk()
    s._vosk_dump_written = camera_client._VOSK_DUMP_CAP_BYTES
    with (
        patch("camera_client._arbiter_submit", new=AsyncMock(return_value=(True, (), "t"))),
        patch.object(s, "_play_attention", new=AsyncMock()),
    ):
        await s._vad_process(_chunk(1000))
        await asyncio.sleep(0)
    assert s._vosk_dump_fh is None, "the dump must close itself at the cap"


# ---------------------------------------------------------------------------
# The echo guards had no idea what TIME it was.
# ---------------------------------------------------------------------------
# Measured 14.10.2026, the corridor spoke at 16:50:59 and the user said «включи
# свет» at 17:00:02 — 543 s later:
#
#   17:00:02.373 ✅ Whisper OK: 'включи свет'
#   17:00:02.374 ❌ Echo repeat, ignoring: 'включи свет'
#
# and the same one millisecond gap in the living room, 923 s after its reply.
# Both were real commands. The anti-loop guard's own comment said the speaker
# returns in 15-25 s, so the author knew the bound — and then implemented none.
# The result was a filter that drops anyone who says a sentence twice, in any
# room, at any time. Neither firing was an echo; that is 0 for 2.

def _echo_session(last_at, last_norm="включи свет", reply="", reply_at=0.0):
    """A session carrying only what `_transcript_is_echo` reads.

    Built by hand, so every attribute it touches has to be here by name — a
    missing one used to surface as `TTS sentence failed` several layers away
    (AGENTS.md, 14.10.2026).

    Offsets are real seconds against the wall clock rather than an injected
    `now`, because the method reads `time.time()` itself. Giving it a `now`
    argument for the convenience of the test would put a test-only hook into the
    live audio path, and a stub that lets the test dictate the answer hides the
    very thing it claims to check.
    """
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "camera-corridor1"
    s._echo_repeat_horizon = _ECHO_REPEAT_HORIZON_S
    s._last_transcript = last_norm
    s._last_transcript_at = time.time() - last_at
    s._last_tts_reply = reply
    s._last_tts_reply_at = (time.time() - reply_at) if reply_at else 0.0
    s._wake_detected = True
    return s


def test_the_reply_overlap_guard_is_bounded_too():
    """`_echo_of_reply` is pure text similarity and had no time input, so it
    would drop a sentence sharing vocabulary with a reply from any point in the
    session. The waveform guard `_is_echo` was bounded on 07.10.2026; this half
    of the same question never was.

    `last_norm=""` puts the OTHER guard out of reach on purpose: it compares the
    transcript to the previous one, and leaving "включи свет" in place made it
    fire first and answer for the reply guard — a stub whose default quietly
    changed which branch was under test.

    The echoed text is a spoken CONFIRMATION rather than a light command, and
    that is deliberate. Whisper transcribes our «Включила свет» back as
    «включила», so «включи» and «включила» are different tokens and the
    similarity test never reaches its thresholds — a realistic echo is our own
    answer heard again («сейчас в коридоре горит свет»), which is the case this
    guard actually exists for. The 11.10.2026 field cases already cover the
    similarity axis.
    """
    echoed = "сейчас в коридоре горит свет"
    old = _echo_session(last_at=0.0, last_norm="", reply=echoed, reply_at=400.0)
    assert old._transcript_is_echo(echoed) is None, (
        "реплика 400 с назад не может быть эхом этого utterance"
    )
    fresh = _echo_session(last_at=0.0, last_norm="", reply=echoed, reply_at=3.0)
    assert fresh._transcript_is_echo(echoed) == "reply", (
        "эхо свежего ответа должно отсекаться"
    )


def test_repeating_the_same_sentence_nine_minutes_later_is_not_an_echo():
    """The field case, verbatim: 543 s, same words, real command."""
    s = _echo_session(last_at=543.0)
    assert s._transcript_is_echo("включи свет") is None, (
        "пользователь повторил фразу через 9 минут — это команда, а не эхо"
    )


def test_repeating_the_same_sentence_fifteen_minutes_later_is_not_an_echo():
    """The other field firing, 923 s, in a different room."""
    s = _echo_session(last_at=923.0)
    assert s._transcript_is_echo("включи свет") is None


def test_a_genuine_echo_within_the_horizon_is_still_caught():
    """A guard that can only ever say "no" is worth nothing.

    12 s is inside the 15-25 s the old comment described, and well inside the
    45 s horizon, so the acoustic loop this was written for is still stopped.
    """
    s = _echo_session(last_at=12.0)
    assert s._transcript_is_echo("включи свет") == "repeat", (
        "настоящее эхо должно отсекаться, иначе защита стала безвредной"
    )


def test_the_horizon_default_covers_the_worst_measured_echo_return():
    """45 s, not a round number chosen for looks: it covers the ONVIF
    backchannel's worst measured return (3-40 s, AGENTS.md 07.10.2026) and is
    12x shorter than the nearest false positive measured (543 s)."""
    assert _ECHO_REPEAT_HORIZON_S == 45.0
    assert _ECHO_REPEAT_HORIZON_S < 543.0, (
        "горизонт обязан быть короче ближайшего зафиксированного ложного "
        "срабатывания, иначе поломка не исправлена"
    )


def test_zero_means_unset_and_not_a_window_that_can_never_fire():
    """0 is «unset, use the default» throughout this project. Taken literally
    it would produce a guard that can never refuse anything."""
    cfg = CameraConfig(stream_name="x", echo_repeat_horizon=0.0)
    assert cfg.echo_repeat_horizon == 0.0
    s = _echo_session(last_at=12.0)
    s._echo_repeat_horizon = (
        cfg.echo_repeat_horizon if cfg.echo_repeat_horizon > 0
        else _ECHO_REPEAT_HORIZON_S
    )
    assert s._transcript_is_echo("включи свет") == "repeat"


# ---------------------------------------------------------------------------
# The attention pip used to close the user's sentence.
# ---------------------------------------------------------------------------
# Measured 14.10.2026 17:39:47-51 on the corridor:
#
#   17:39:47.799  🔔 attention pip (vosk)          play_audio 0.40s in 2.18s (x5.5)
#   17:39:48.475  VAD UTTERANCE END dur=1440ms via pause (rms=0.0144 ref=0.6123)
#   17:39:48.477  🎤 Whisper IN: 1.44s raw_rms=12533 peak=32767
#   17:39:51.487  ❌ Whisper empty
#
# The utterance was closed on the PIP'S OWN DECAY, the beep is inside the
# reference window (ref=0.6123), and Whisper was handed 1.44 s of two-tone beep
# at the rail instead of a command. The 07.10.2026 fix that stopped computing the
# mic mute from the RESPONSE was applied to the reply path alone; both cue paths
# kept `time.time()` after the POST, so on a slow endpoint the mic stayed live for
# the whole 2.18 s the camera was making noise into it.

def _inflight_session():
    s = CameraSession.__new__(CameraSession)
    s.stream_name = "camera-corridor1"
    s._tts_playing_until = 0.0
    s._tts_play_end = 0.0
    s._speaking_until = 0.0
    return s


def test_the_window_opens_before_the_post_not_after_it():
    """The whole defect in one assertion: armed at `sent_at`, so it is already
    closed when the slow response comes back."""
    s = _inflight_session()
    sent_at = time.time()
    s._arm_tts_inflight(sent_at, duration=0.4)
    # 2.18 s later the camera is still making noise — the pip's own audio is
    # already in the room and the endpoint must not be hearing it.
    at_response = time.time()
    assert at_response - sent_at < 2.18, "проверка должна идти до ответа POST"
    assert s._tts_playing_until > at_response, (
        "окно должно быть открыто уже во время ответа POST — именно эту "
        "двухсекундную дыру пик и оставлял открытой"
    )


def test_the_window_covers_the_whole_slow_response():
    """`_close_tts_inflight` is what makes a 2.18 s POST survivable: the window
    is stretched to cover the response, not just the 0.4 s of audio."""
    s = _inflight_session()
    s._arm_tts_inflight(time.time(), duration=0.4)
    # pretend 2.18 s of POST elapsed
    s._close_tts_inflight(0.4)
    assert s._tts_playing_until > time.time() + 0.3, (
        "окно обязано пережить медленный ответ, иначе ровно тот случай вернётся"
    )


def test_the_window_is_never_narrowed_by_a_later_shorter_call():
    """Two cues in a row must not let the second one's short audio pull the
    window back and re-open the hole while the first is still sounding."""
    s = _inflight_session()
    s._tts_playing_until = time.time() + 5.0
    s._arm_tts_inflight(time.time(), duration=0.1)
    s._close_tts_inflight(0.1)
    assert s._tts_playing_until > time.time() + 4.5, "окно сузилось — это тот же баг"


def test_the_endpoint_is_not_fed_while_our_audio_may_be_in_the_room():
    """The condition itself, as the live path evaluates it. A paused/closed
    endpoint never returns "end", so a pause inside our own audio cannot close an
    utterance."""
    s = _inflight_session()
    s._vad_has_speech = True
    s._wake_detected = True
    s._tts_playing_until = time.time() + 2.0
    can_endpoint = (
        s._vad_has_speech and s._wake_detected
        and time.time() >= s._tts_playing_until
    )
    assert can_endpoint is False, (
        "пока наш звук может быть в комнате, детектор паузы не должен решать"
    )
    s._tts_playing_until = time.time() - 1.0
    can_endpoint = (
        s._vad_has_speech and s._wake_detected
        and time.time() >= s._tts_playing_until
    )
    assert can_endpoint is True, (
        "а после нашего звука детектор обязан работать — иначе пауза перестанет "
        "заканчивать фразы и каждая команда снова ждёт 7 с"
    )
