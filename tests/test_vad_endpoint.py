"""Level-based utterance endpointing — the 7 s dead wait.

Field case 04.10.2026: wake at 19:20:10.1, Whisper at 19:20:17.3. 7.2 s, of
which the user's «выключи свет» was the first 1.5. The cause is that an
utterance is ended by VAD SILENCE, and the living room's Silero never reports
silence — 0 'speech=False' in 15 live minutes, with every
'VAD rms=... speech=True' line sitting above 'consec=250'. So every command ran
to the 7 s duration cap.

Lowering the cap is NOT the fix: it is load-bearing for long commands.
«я просил включить следующую серию черного зеркала» transcribes correctly at
7.04 s and is chopped at 3.5 s. The endpoint therefore reads the LEVEL
envelope, which dips between words whatever the VAD thinks.

The constraint that outranks latency: if the reference cannot be established
(speech quieter than the background), it must report no pause at all and fall
back to the cap. Slower is recoverable; a chopped command is a wrong command.
"""

import inspect
import os

from camera_client import (
    CameraConfig,
    CameraSession,
    _NoiseFloor,
    _PauseEndpoint,
)


# 160 ms frames, i.e. 6.25/s — the hop _vad_process actually runs at.
SPEECH = [0.020, 0.024, 0.022, 0.026, 0.023, 0.025, 0.021, 0.024]
# A real inter-word gap: below 0.55 * the 0.020-0.026 speech reference.
PAUSE = [0.008, 0.006, 0.007, 0.005, 0.006, 0.004, 0.005, 0.003]

# Six pause frames = 0.96 s, which is what run_frames defaults to.
END_AT = 5


def _feed(ep, levels):
    return [ep.feed(v)[0] for v in levels]


def _first_end(ep, levels):
    """Index of the first "end" verdict, or None.

    Asserting on the INDEX rather than on the last element matters: the detector
    fires on the sixth dip frame and then keeps classifying, so a test that only
    looks at the final element passes even when the detector fires on the wrong
    frame — and fails when it fires correctly but early.
    """
    for i, v in enumerate(levels):
        if ep.feed(v)[0] == "end":
            return i
    return None


def test_a_command_ends_on_the_gap_after_it_not_on_the_cap():
    ep = _PauseEndpoint()
    assert _feed(ep, SPEECH) == ["speech"] * len(SPEECH)
    assert _first_end(ep, PAUSE) == END_AT


def test_the_gap_right_after_the_wake_word_does_not_end_the_utterance():
    """«компьютер» is one continuous word; the pause after it is a gap between
    words, not the end of a sentence. Dispatching here sends a bare wake word to
    Whisper and wastes the command — the failure behind the bare «Да?».
    """
    ep = _PauseEndpoint()
    wake = [0.020, 0.024, 0.023]        # ~0.5 s of «компьютер»
    assert _feed(ep, wake) == ["speech"] * 3
    # A FULL pause run, but only 3 speech frames so far: must not end.
    assert _first_end(ep, PAUSE) is None, "a bare wake word was dispatched"
    # Once real speech has been seen, a pause does end the utterance.
    assert _first_end(ep, SPEECH) is None
    assert _first_end(ep, PAUSE) == END_AT


def test_a_short_gap_between_words_is_not_the_end():
    """«включи свет и поставь музыку» has real pauses inside it: three dip
    frames must not commit the utterance."""
    ep = _PauseEndpoint()
    _feed(ep, SPEECH)
    assert _first_end(ep, [0.008, 0.007, 0.006]) is None, "3 dip frames ended it"
    assert _first_end(ep, SPEECH) is None, "speech did not clear the dip run"
    assert _first_end(ep, PAUSE) == END_AT


def test_the_endpoint_is_what_removes_the_wait_not_the_cap():
    """The number that matters: how much earlier than the cap the command goes
    out. Eight speech frames (1.28 s) plus the dip run is ~2.2 s, against the
    7 s cap that every command used to wait out."""
    ep = _PauseEndpoint()
    _feed(ep, SPEECH)
    idx = _first_end(ep, PAUSE)
    assert idx is not None
    frames_to_end = len(SPEECH) + idx + 1
    assert frames_to_end * 0.160 < 3.0, "endpointing is not buying anything"


def test_speech_quieter_than_the_background_never_ends_the_utterance():
    """The safe direction: no reference means no pause, so the duration cap
    still fires. This is why the reference is a trailing percentile and not an
    absolute floor — with the television on there is no floor to measure.
    """
    ep = _PauseEndpoint()
    _feed(ep, [0.050] * 20)                       # steady television
    assert _first_end(ep, [0.030] * 30) is None    # the user, well under it


def test_one_transient_cannot_redefine_the_reference():
    """A door slam must not make the rest of the sentence look like a pause —
    hence the 80th percentile and not the maximum."""
    ep = _PauseEndpoint()
    _feed(ep, SPEECH)
    _feed(ep, [0.400])                            # the slam
    assert _first_end(ep, SPEECH) is None


def test_the_detail_carries_the_numbers_the_verdict_was_made_from():
    """Without these the endpoint can only be tuned by guessing."""
    ep = _PauseEndpoint()
    _feed(ep, SPEECH)
    _verdict, detail = ep.feed(PAUSE[0])
    for key in ("rms=", "ref=", "floor=", "run=", "speech="):
        assert key in detail, key + " missing from the endpoint detail"


def test_silence_is_never_counted_as_speech():
    """The regression the windowed reference caused.

    With the speech floor taken from the TRAILING PERCENTILE, a pause long
    enough to fill the 12-frame window collapses that percentile to the room
    floor, the pause test stops matching, and every remaining quiet frame is
    counted as speech. A bare «компьютер» (4 frames) plus a 30-frame pause drove
    _speech_frames to 50 on silence alone — and min_speech_frames exists exactly
    to exclude that case, so the guard was not guarding anything.

    The floor is utterance-scoped and frozen, so silence adds nothing.
    """
    ep = _PauseEndpoint()
    _feed(ep, [0.020, 0.024, 0.023, 0.026])    # anchor established here
    before = ep._speech_frames
    _feed(ep, [0.008] * 30)
    assert ep._speech_frames == before, (
        "silence was counted as speech: "
        f"{before} -> {ep._speech_frames}"
    )
    assert _first_end(ep, [0.008] * 20) is None


def test_a_transient_after_the_anchor_is_set_cannot_raise_it():
    """The anchor is frozen on purpose: a door slam must not redefine "speech"
    for the rest of the sentence. Measured consequence of not freezing it — the
    slam sets the floor to 0.2 and the user can never register again."""
    ep = _PauseEndpoint()
    _feed(ep, [0.020, 0.024, 0.023, 0.026])
    anchor = ep._anchor
    _feed(ep, [0.400])
    assert ep._anchor == anchor, "a door slam moved the speech floor"


def test_reset_clears_the_dip_run_between_utterances():
    ep = _PauseEndpoint()
    _feed(ep, SPEECH)
    _feed(ep, PAUSE[:END_AT])                     # one frame short of ending
    ep.reset()
    assert _first_end(ep, PAUSE[:END_AT]) is None, "state leaked into the next one"


def test_the_flag_is_off_by_default():
    """It changes WHEN a command is dispatched in a live audio path."""
    assert CameraConfig(stream_name="livingroom").pause_endpoint is False
    assert CameraConfig(stream_name="x", pause_endpoint=True).pause_endpoint is True


def test_the_env_wiring_exists_and_defaults_to_off():
    """The flag is useless if nothing reads it, and dangerous if the default is
    anything but off — so pin BOTH in main.py's CameraConfig construction."""
    main_py = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
    )
    with open(main_py, encoding="utf-8") as fh:
        src = fh.read()
    assert "CAMERA_PAUSE_ENDPOINT_{name.upper()}" in src, (
        "the per-room env name is not wired"
    )
    assert 'os.getenv("CAMERA_PAUSE_ENDPOINT", "false")' in src, (
        "the global default must be off"
    )
    assert "pause_endpoint=" in src, "CameraConfig is never given the flag"

    # Tuning must be reachable by env: the logged rms/ref/floor exist to be
    # tuned from, and tuning that costs a rebuild per iteration is not tuning.
    for _name in ("CAMERA_PAUSE_RATIO", "CAMERA_PAUSE_RUN_FRAMES",
                  "CAMERA_PAUSE_MIN_SPEECH_FRAMES"):
        assert _name in src, _name + " is not overridable per room"


def test_zero_means_default_not_zero_threshold():
    """A half-filled env override must not silently disable a threshold.

    'pause_ratio=0.0' is how CameraConfig says "unset". If __init__ took it
    literally, every ratio test rms < ref * 0 would fail, no pause would ever be
    reported, and the utterance would fall back to the 7 s cap — which looks
    exactly like "the endpoint does not work".
    """
    ep = _PauseEndpoint(ratio=0.0, run_frames=0, min_speech_frames=0)
    assert ep.ratio == 0.55
    assert ep.run_frames == 6
    assert ep.min_speech_frames == 5
    _feed(ep, SPEECH)
    assert _first_end(ep, PAUSE) == END_AT, "a zeroed threshold broke detection"


def test_env_values_do_reach_the_detector():
    ep = _PauseEndpoint(ratio=0.9, run_frames=3, min_speech_frames=2)
    assert ep.ratio == 0.9 and ep.run_frames == 3 and ep.min_speech_frames == 2
    _feed(ep, SPEECH)
    assert _first_end(ep, PAUSE) == 2, "the override was not honoured"


def test_all_three_terminators_go_through_one_path():
    """Silence, pause and cap must not each inline their own reset block.

    The duplication is not hypothetical: the three copies had already drifted,
    which is how one path ends up resetting state the others do not. (The only
    difference was _vad_start_time — dead state, written three times and read
    nowhere; removed rather than propagated.)
    """
    src = inspect.getsource(CameraSession._vad_process)
    assert src.count("self._vad_speech_buf.clear()") == 0, (
        "_vad_process must not clear the buffer itself"
    )
    assert src.count("self._process_utterance(") == 0, (
        "_vad_process must not dispatch itself"
    )
    assert src.count("self._end_utterance(") == 3, "silence + cap + pause"


def test_every_end_is_logged_with_its_reason():
    fn = inspect.getsource(CameraSession._end_utterance)
    assert "via {reason}" in fn
    assert "_end_reason" in fn, (
        "the per-boot tally is what answers whether the endpoint is working or "
        "everything is still hitting the 7 s cap"
    )


# --- the room's noise floor, measured 05.10.2026 ----------------------------
#
# The trailing-window reference could not find a pause in this room at all: 4 of
# 4 real commands ran the full 7 s cap while the command itself was over in
# 1.28-1.44 s. Two independent reasons, both visible in the envelope below.


def test_the_noise_floor_is_not_defined_by_a_transient():
    """The attention pip is 0.59 rms against a 0.010 floor here.

    Seeded from the first frame, or from the mean, the pip BECOMES the floor and
    every quiet frame then reads as speech.
    """
    nf = _NoiseFloor()
    for rms in [0.59, 0.018, 0.012, 0.088, 0.038, 0.017] + [0.0108] * 6:
        nf.feed(rms)
    assert 0.008 < nf.value < 0.013, f"floor={nf.value:.4f} — a transient won"


def test_the_noise_floor_moves_down_fast_and_up_slow():
    nf = _NoiseFloor()
    for _ in range(12):
        nf.feed(0.010)
    quiet = nf.value
    nf.feed(0.005)
    assert nf.value < quiet, "floor must fall when the room goes quiet"
    dropped = quiet - nf.value
    for _ in range(20):
        nf.feed(0.010)
    assert nf.value - (quiet - dropped) < dropped, (
        "floor must rise slowly, or one burst of noise redefines the room"
    )


# The measured 160 ms rms envelopes of two real commands, saved by the gateway in
# /tmp/utterances on 05.10.2026 18:32 UTC. Frames 0-1 are the attention pip
# (0.59 / 0.48 rms against a 0.010 room floor), the words follow, then the room.
# `включи свет` is over at frame 10 and `выключи свет` at frame 8.
_REAL_COMMAND_RMS = [
    0.5923, 0.4781, 0.0181, 0.0113, 0.0125, 0.0114, 0.0879, 0.0719,
    0.0377, 0.0423, 0.0174, 0.0111, 0.0108, 0.0123, 0.0113, 0.0100,
    0.0107, 0.0108, 0.0114, 0.0109, 0.0108, 0.0104, 0.0102, 0.0117,
]
_SECOND_COMMAND_RMS = [
    0.5954, 0.4810, 0.0277, 0.0509, 0.0601, 0.0420, 0.0266, 0.0311,
    0.0175, 0.0109, 0.0103, 0.0099, 0.0121, 0.0119, 0.0125, 0.0112,
    0.0109, 0.0104, 0.0119, 0.0115, 0.0121, 0.0123, 0.0099, 0.0115,
]


# The room before the command, so the floor is established the way the live loop
# has it: tracked on every frame for minutes before anyone speaks. Without this
# the first 12 frames are spent SEEDING and cannot be classified at all, which is
# an artefact of replaying an utterance in isolation, not of the room.
_ROOM = [0.0110, 0.0108, 0.0113, 0.0105, 0.0109, 0.0111, 0.0107, 0.0104,
         0.0112, 0.0106, 0.0109, 0.0107]


def _fire(rms_list, mult=2.5):
    noise = _NoiseFloor(mult=mult)
    ep = _PauseEndpoint(noise_mult=mult, noise=noise)
    for v in _ROOM:
        noise.feed(v)
    for i, v in enumerate(rms_list):
        noise.feed(v)
        if ep.feed(v)[0] == "end":
            return i
    return None


def test_a_real_command_ends_on_its_pause_and_not_on_the_cap():
    """Both commands finished long before the 7 s cap: 2.40 s and 2.08 s, i.e.
    0.96 s after the last word, which is the run_frames confirmation window."""
    first, second = _fire(_REAL_COMMAND_RMS), _fire(_SECOND_COMMAND_RMS)
    assert first is not None, "the pause was never found"
    assert first * 0.16 == 2.40, f"ended at {first * 0.16:.2f}s, not 2.40s"
    assert second * 0.16 == 2.08, f"ended at {second * 0.16:.2f}s, not 2.08s"
    # Never before the last word: the commands end at frames 10 and 8.
    assert first * 0.16 > 1.60, "cut the command short"
    assert second * 0.16 > 1.28, "cut the command short"


def test_the_windowed_reference_still_cannot_find_that_pause():
    """Why the noise floor was needed, kept as a guard: same envelopes, shipped
    detector. The reference is the pip for a second, then decays onto the room,
    so silence is never below a fraction of it."""
    for rms in (_REAL_COMMAND_RMS, _SECOND_COMMAND_RMS):
        ep = _PauseEndpoint()
        for v in rms:
            assert ep.feed(v)[0] != "end", (
                "the windowed reference now finds this pause; if that holds on "
                "real audio the noise-floor path should be revisited rather than "
                "kept as the default"
            )


def test_the_room_floor_is_what_it_measures_and_not_the_pip():
    noise = _NoiseFloor(mult=2.5)
    for v in _REAL_COMMAND_RMS:
        noise.feed(v)
    assert 0.008 < noise.value < 0.013, f"floor={noise.value:.4f} — a transient won"


def test_continuous_sound_still_falls_back_to_the_cap():
    """Our own TTS echo, measured at the same time: an utterance the detector
    must NOT cut. The safe direction is the cap — slower, never chopped.

    This is also the test that caught the floor being raised by the speech it
    was supposed to measure against: 7 s at 0.04-0.06 with no gap ended early
    because the floor had climbed to meet it.
    """
    import numpy as np

    noise = _NoiseFloor(mult=2.5)
    ep = _PauseEndpoint(noise_mult=2.5, noise=noise)
    for _ in range(12):
        noise.feed(0.010)
    rng = np.random.default_rng(7)
    for _ in range(44):  # 7 s at speech level with no gap at all
        v = 0.04 + 0.02 * float(rng.random())
        noise.feed(v)
        assert ep.feed(v)[0] != "end", "cut an utterance that never paused"
    assert noise.value < 0.02, (
        f"floor climbed to {noise.value:.4f} on speech alone — it must only "
        "track frames it considers noise"
    )


def test_a_raised_background_is_followed_so_the_room_still_works():
    """The slow upward branch exists for this: a vacuum cleaner or a hood lifts
    the floor over seconds and the detector keeps working instead of reading all
    of it as speech."""
    noise = _NoiseFloor(mult=2.5)
    for _ in range(12):
        noise.feed(0.010)
    for _ in range(400):  # ~64 s of background at 0.018, still below 2.5x
        noise.feed(0.018)
    assert 0.015 < noise.value < 0.021, f"floor={noise.value:.4f}"
    ep = _PauseEndpoint(noise_mult=2.5, noise=noise)
    fired = None
    # Four frames of voice, then a gap of six — exactly what run_frames demands.
    for i in range(40):
        v = 0.055 if i % 10 < 4 else 0.018
        noise.feed(v)
        if ep.feed(v)[0] == "end":
            fired = i
            break
    assert fired is not None, "lost the pause once the room got noisier"
    # Not frame 10: that gap came after only 4 speech frames, and
    # min_speech_frames is 5 — the guard that stops a bare «компьютер» plus a
    # pause from dispatching an empty command. The second gap satisfies it.
    assert fired == 19, f"ended at frame {fired}, expected 19"
    assert 10 < fired, "the min_speech_frames guard did not hold"


def test_no_floor_means_no_pause():
    """Frames arrive before the floor is established; that must not read as a
    pause, and must certainly not end the utterance."""
    ep = _PauseEndpoint(noise_mult=2.5, noise=_NoiseFloor())
    verdicts = [ep.feed(0.0001)[0] for _ in range(10)]
    assert all(v == "speech" for v in verdicts), verdicts
