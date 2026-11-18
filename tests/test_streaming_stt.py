"""Tests for LocalAgreement-2 streaming STT.

The invariant that matters: a committed prefix is stable across two
consecutive windows, and a disagreement ROLLS IT BACK. A word that Whisper
rewrites on the next window must never reach the router.
"""

import asyncio

import pytest

import streaming_stt as ss


def test_nothing_is_committed_from_one_hypothesis():
    la = ss.LocalAgreement(n=2)
    text, changed = la.add("включи свет в гостиной")
    assert text == "" and changed is False


def test_two_identical_hypotheses_commit_the_prefix():
    la = ss.LocalAgreement(n=2)
    la.add("включи свет в гостиной")
    text, changed = la.add("включи свет в гостиной")
    assert changed is True
    assert text == "включи свет в гостиной"


def test_a_growing_transcript_commits_only_the_stable_prefix():
    la = ss.LocalAgreement(n=2)
    la.add("включи свет")
    text, _ = la.add("включи свет в")
    assert text == "включи свет"
    la.add("включи свет в гостиной")
    text, _ = la.add("включи свет в гостиной")
    assert text == "включи свет в гостиной"


def test_disagreement_rolls_the_prefix_back():
    la = ss.LocalAgreement(n=2)
    la.add("включи свет в спальне")
    text, _ = la.add("включи свет в спальне")
    assert text == "включи свет в спальне"
    text, _ = la.add("включи свет в гостиной")   # one disagreeing window
    assert text == "включи свет в"               # rolled back, no new word
    text, changed = la.add("включи свет в гостиной")
    # Only after TWO agreeing windows may the new ending be committed.
    assert text == "включи свет в гостиной"
    assert changed is True


def test_a_rewritten_word_is_never_committed_from_a_single_window():
    la = ss.LocalAgreement(n=2)
    la.add("выключи свет")
    text, changed = la.add("выключи кофеварку")
    assert text == "выключи"
    assert changed is True


def test_reset_clears_both_hypotheses_and_commit():
    la = ss.LocalAgreement(n=2)
    la.add("привет")
    la.add("привет")
    assert la.text == "привет"
    la.reset()
    assert la.text == ""
    assert la.hypotheses == []


@pytest.mark.asyncio
async def test_streaming_transcriber_uses_overlapping_windows():
    script = ["включи", "включи свет", "включи свет", "включи свет",
              "включи свет в гостиной"]

    calls = []

    async def fake_transcribe(pcm: bytes, rms: float) -> str:
        calls.append(len(pcm))
        return script[min(len(calls) - 1, len(script) - 1)]

    st = ss.StreamingTranscriber(fake_transcribe, window_s=2.0, step_s=0.5)
    step = st.step_bytes
    for i in range(8):                 # 8 steps = 4 s of audio
        await st.feed(b"\x01\x02" * (step // 2), 1000.0, now=float(i))
    assert st.windows == 5
    assert calls == [st.window_bytes] * 5
    assert st.committed == "включи свет"


@pytest.mark.asyncio
async def test_a_short_utterance_never_reaches_whisper():
    called = []

    async def fake_transcribe(pcm, rms):
        called.append(pcm)
        return "тест"

    st = ss.StreamingTranscriber(fake_transcribe)
    await st.feed(b"\x01\x02" * 100, 1000.0, now=0.0)   # 200 bytes
    assert called == [] and st.committed == ""


def test_window_must_cover_n_steps():
    with pytest.raises(ValueError):
        ss.StreamingTranscriber(lambda p, r: asyncio.sleep(0), window_s=0.5, step_s=0.5, n=2)
