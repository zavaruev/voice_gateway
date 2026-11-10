"""Emotion -> speed: the tag travels as DATA, the speed reaches the synthesiser.

Chain under test (each link independent, no network):
  router chat_proxy  captures the tag from the sentence head (strips it from
                     the text) -> SSE sentence event carries `emotion`;
  backends           maps emotion -> speed (sad/angry -> 0.9, everything
                     else 1.0) and pushes a TTSentence(str) carrying .speed;
  camera_client      _tts_fetch adds "speed" to the payload ONLY when it
                     differs from 1.0 (edge_tts fallback payloads stay
                     byte-identical) and reads .speed off the queue item.

The contract that must not drift: TTSentence IS a str — every existing
consumer (players, follow-up question detection, history) keeps working.
"""

import asyncio
import os
import sys
from unittest import mock

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import backends  # noqa: E402
from backends import CascadeBackend, EMOTION_SPEED, TTSentence  # noqa: E402

_ROUTER = os.path.abspath(os.path.join(_ROOT, "services", "jev-router"))
sys.path.insert(0, _ROUTER)

import chat_proxy  # noqa: E402


# --------------------------------------------------------------------------
# TTSentence + the speed map
# --------------------------------------------------------------------------

def test_ttsentence_is_a_str_and_travels_as_one():
    s = TTSentence("Мне очень жаль.", 0.9)
    assert isinstance(s, str), "TTSentence must satisfy every str consumer"
    assert s == "Мне очень жаль."
    assert s.speed == 0.9
    assert (s + " Что случилось?").endswith("Что случилось?")
    assert s.upper() == "МНЕ ОЧЕНЬ ЖАЛЬ."


def test_ttsentence_defaults_to_neutral_speed():
    assert TTSentence("Просто предложение.").speed == 1.0


def test_speed_map_has_only_the_approved_pairings():
    """0.9 exists for exactly the two emotions the user approved
    (10.11.2026); everything else must stay at 1.0 — a speed with no
    verdict behind it must not change how the house sounds."""
    assert EMOTION_SPEED == {"sad": 0.9, "angry": 0.9}


def test_speed_for_is_case_insensitive_and_fails_neutral():
    assert backends._speed_for("sad") == 0.9
    assert backends._speed_for("SAD") == 0.9
    assert backends._speed_for("angry") == 0.9
    for emo in ("", None, "happy", "neutral", "surprised", "thinking", "grief"):
        assert backends._speed_for(emo) == 1.0, emo


def test_with_speed_wraps_only_non_neutral():
    plain = backends._with_speed("Текст.", 1.0)
    assert not isinstance(plain, TTSentence), "neutral queue items stay plain str"
    slow = backends._with_speed("Текст.", 0.9)
    assert isinstance(slow, TTSentence) and slow.speed == 0.9


# --------------------------------------------------------------------------
# Router: tag capture
# --------------------------------------------------------------------------

def test_head_emotion_captures_the_last_tag():
    assert chat_proxy._head_emotion("[sad] Мне очень жаль.") == "sad"
    assert chat_proxy._head_emotion("[happy] Отлично!") == "happy"


def test_head_emotion_last_tag_wins():
    assert chat_proxy._head_emotion("[sad] А. [happy] Б.") == "happy"


def test_head_emotion_absent_tag_is_empty():
    assert chat_proxy._head_emotion("Без тега вообще.") == ""


def test_speakable_still_strips_the_emotion_tag():
    """The tag must never be spoken — capture and strip are separate steps."""
    s = chat_proxy._speakable("[sad] Мне очень жаль. Что случилось?")
    assert "sad" not in s and "[" not in s
    assert s == "Мне очень жаль. Что случилось?"


# --------------------------------------------------------------------------
# CascadeBackend: the SSE `emotion` field becomes a TTSentence
# --------------------------------------------------------------------------

def _ev(obj: dict) -> bytes:
    import json
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n"


class _FakeContent:
    def __init__(self, events):
        self._lines = [_ev(e) for e in events] + [b"data: [DONE]\n"]

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line

        return gen()


class _FakeResp:
    status = 200

    def __init__(self, events):
        self.content = _FakeContent(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return "boom"


class _FakePost:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return _FakeResp(self._events)

    async def __aexit__(self, *a):
        return False


class _FakeSess:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, *a, **k):
        return _FakePost(self._events)


async def _run(events):
    q = asyncio.Queue()
    be = CascadeBackend(router_url="http://x", ack_delay=3.0)
    fake = _FakeSess(events)
    with mock.patch("backends.aiohttp.ClientSession", return_value=fake):
        await be.generate_response("тест", "sess", "test", q)
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    assert out[-1] is None, "exactly one None sentinel is the queue contract"
    return out[:-1]


def test_cascade_sad_sentence_arrives_slow():
    events = [
        {"type": "route", "route": "general_qa", "confidence": 0.9},
        {"type": "sentence", "text": "Мне очень жаль.", "emotion": "sad"},
        {"type": "sentence", "text": "Обычное предложение без эмоции."},
        {"type": "done", "route": "general_qa", "elapsed": 1.0},
    ]
    out = asyncio.run(_run(events))
    slow = out[0]
    assert isinstance(slow, TTSentence), f"sad must carry the subclass: {slow!r}"
    assert slow.speed == 0.9
    assert slow == "Мне очень жаль.", "text must be unchanged (is-a str)"
    neutral = out[1]
    assert neutral == "Обычное предложение без эмоции."
    assert not isinstance(neutral, TTSentence), "untagged stays plain str"


def test_cascade_angry_sentence_arrives_slow():
    events = [
        {"type": "sentence", "text": "Это недопустимо!", "emotion": "angry"},
        {"type": "done", "route": "general_qa", "elapsed": 1.0},
    ]
    out = asyncio.run(_run(events))
    assert isinstance(out[0], TTSentence) and out[0].speed == 0.9


def test_cascade_unknown_emotion_stays_neutral():
    events = [
        {"type": "sentence", "text": "Удивительно!", "emotion": "surprised"},
        {"type": "done", "route": "general_qa", "elapsed": 1.0},
    ]
    out = asyncio.run(_run(events))
    assert out[0] == "Удивительно!"
    assert not isinstance(out[0], TTSentence)


# --------------------------------------------------------------------------
# camera_client: payload + prefetch wiring
#
# camera_client imports PyAV at module level, which exists only in the
# image (the host pytest run never collects test_camera_client.py for the
# same reason); these two tests run there and in CI.
# --------------------------------------------------------------------------

try:
    import camera_client  # noqa: F401
    _HAVE_AV = True
except ModuleNotFoundError:
    _HAVE_AV = False

needs_av = pytest.mark.skipif(
    not _HAVE_AV, reason="camera_client imports PyAV, present only in the image"
)

_RAW_PCM = (b"\x01\x00" * 4800)


class _RecResp:
    status = 200

    def __init__(self, raw):
        self._raw = raw

    async def read(self):
        return self._raw


class _RecPost:
    def __init__(self, raw):
        self._raw = raw

    async def __aenter__(self):
        return _RecResp(self._raw)

    async def __aexit__(self, *_a):
        return False


class _RecSession:
    """Records every json payload posted; answers with a canned mp3."""

    def __init__(self, raw):
        self._raw = raw
        self.payloads = []

    def post(self, _url, json=None, **_k):
        self.payloads.append(json)
        return _RecPost(self._raw)


class _FakeSeg:
    def __init__(self, raw):
        self.raw_data = raw

    def set_frame_rate(self, _r):
        return self

    def set_channels(self, _c):
        return self

    def set_sample_width(self, _w):
        return self


def _tts_session():
    from camera_client import CameraSession, _TTS_TARGET_PEAK_DEFAULT

    s = CameraSession.__new__(CameraSession)
    s.stream_name = "cam"
    s.tts_api_key = ""
    s.tts_url = "http://tts"
    s.tts_voice = "v"
    s._tts_target_peak = _TTS_TARGET_PEAK_DEFAULT
    return s


@needs_av
def test_payload_speed_key_only_when_not_neutral():
    """Coqui gets speed only for a real slow-down; edge_tts, the fallback
    endpoint, must keep the EXACT payload it always got."""
    from unittest.mock import patch

    async def body():
        s = _tts_session()
        sess = _RecSession(_RAW_PCM)
        s.http_session = sess
        with patch("camera_client.AudioSegment.from_file",
                   return_value=_FakeSeg(_RAW_PCM)):
            await s._tts_fetch("Привет.")
            await s._tts_fetch("Мне очень жаль.", 0.9)
            await s._tts_fetch("Ноль скорости.", 0)
        return sess.payloads

    base = {"model": "tts-1", "input": "Привет.", "voice": "v",
            "response_format": "mp3"}
    payloads = asyncio.run(body())
    assert payloads[0] == base, "neutral payload must be byte-identical"
    assert "speed" not in payloads[0]
    assert payloads[1].get("speed") == 0.9
    assert "speed" not in payloads[2], "0 is 'unset', not a speed"


@needs_av
def test_prefetch_reads_the_speed_off_the_queue_item():
    """_prefetch_next is the synthesis path for every prefetched sentence —
    the speed must travel from the queue item into _tts_fetch."""

    async def body():
        s = _tts_session()
        seen = []

        async def fake_fetch(text, speed=1.0):
            seen.append((text, speed))
            return b"pcm"

        s._tts_fetch = fake_fetch
        q = asyncio.Queue()
        await q.put(TTSentence("Мне очень жаль.", 0.9))
        sent, pcm = await s._prefetch_next(q)
        assert (sent, pcm) == ("Мне очень жаль.", b"pcm")
        assert seen == [("Мне очень жаль.", 0.9)]
        # A plain str (ack, apology) reads as neutral through getattr.
        await q.put(CascadeBackend.ACK_COMPLEX)
        await s._prefetch_next(q)
        assert seen[-1] == (CascadeBackend.ACK_COMPLEX, 1.0)

    asyncio.run(body())
