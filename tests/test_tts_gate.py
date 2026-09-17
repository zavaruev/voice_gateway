"""TTS gate: internal monologue must not reach speech; real answers kept whole."""
import asyncio
import json
from unittest import mock

import pytest

import backends
from backends import HermesBackend


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield ("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n").encode()
            yield b"data: [DONE]\n"

        return gen()


class _FakeResp:
    status = 200

    def __init__(self, chunks):
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakePost:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return _FakeResp(self._chunks)

    async def __aexit__(self, *a):
        return False


class _FakeSess:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, *a, **k):
        return _FakePost(self._chunks)


async def _run(chunks):
    q = asyncio.Queue()
    be = HermesBackend(url="http://x", api_key="")
    with mock.patch("backends.aiohttp.ClientSession", return_value=_FakeSess(chunks)):
        await be.generate_response("погода", "sess", "test", q)
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return [x for x in out if x is not None]


def test_monologue_dropped_tagged_answer_kept():
    chunks = [
        "fullerene? ",
        "No that's premature. ",
        "Need to execute code to query searxng. ",
        "[neutral] Москва, 16.09: облачно с прояснениями, +10. ",
        "Днем до +19, ясно. ",
    ]
    out = asyncio.run(_run(chunks))
    bad = [s for s in out if any(w in s for w in ["fullerene", "premature", "execute code", "searxng"])]
    good = [s for s in out if "Москва" in s or "ясно" in s]
    assert not bad, f"monologue leaked: {bad}"
    assert good, "real answer lost!"


def test_untagged_reply_still_fully_spoken():
    """Fail-safe: a reply with no emotion tag at all must not be lost."""
    chunks = ["Москва солнечная. ", "Днем до +19, ясно. "]
    out = asyncio.run(_run(chunks))
    joined = " ".join(out)
    assert "Москва" in joined and "ясно" in joined, f"content lost: {out}"


def test_fail_open_streams_without_tag():
    """With _TAG_WAIT_S=0 the gate releases immediately: untagged
    sentences stream instead of buffering to the very end."""
    chunks = ["Первое предложение без тега. ", "Второе без тега. "]
    with mock.patch.object(backends, "_TAG_WAIT_S", 0):
        out = asyncio.run(_run(chunks))
    assert len(out) == 2, f"expected streaming sentences, got: {out}"


def test_abbreviations_not_torn():
    """мм рт. ст. must survive as one utterance, not 'рт.' + 'ст.'."""
    chunks = ["Давление 748 мм рт. ст. Ветер северный. "]
    out = asyncio.run(_run(chunks))
    assert not any(s.strip() == "ст." for s in out), f"torn abbreviation: {out}"
    assert any("рт. ст." in s for s in out), f"abbreviation broken: {out}"
