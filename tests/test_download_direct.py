"""Direct download path: parse the spoken request, call search-and-download,
and speak the confirmation at the moment the torrent is added.

Field case 27.11.2026 13:10 MSK: «скачай восьмой сезон Симпсонов» reached
Hermes, which added S08 at 13:13:56 while the room had already been told
«Простите». The user's decision: never announce completion; say
«Скачивается <название>» when it is found and added, then end the turn.
28.11.2026: an IMMEDIATE «Пытаюсь скачать.» was added so the 2-4 minute
browser search is not silent; the factual sentence still waits for the add.
"""
import asyncio
import inspect
import json
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

from download import (  # noqa: E402
    DownloadRequest,
    confirmation_sentence,
    parse_request,
)


def test_the_field_phrase_parses():
    assert parse_request("скачай восьмой сезон Симпсонов") == DownloadRequest(
        query="Симпсонов", kind="series", season=8
    )


def test_a_wake_word_prefix_is_stripped():
    assert parse_request("компьютер скачай восьмой сезон Симпсонов") == DownloadRequest(
        query="Симпсонов", kind="series", season=8
    )


def test_a_conjugated_download_verb_is_still_a_download():
    """Field case 28.11.2026 12:27 MSK: Whisper wrote «скачает» (future
    tense) and the turn was called suspect STT instead of a download.
    Verb conjugation must not change the route.
    """
    from resolver import is_download_request

    assert is_download_request("скачает третий сезон Симпсонов")
    assert parse_request("скачает третий сезон Симпсонов") == DownloadRequest(
        query="Симпсонов", kind="series", season=3
    )
    assert parse_request("загрузил третий сезон Симпсонов") is None


def test_a_past_tense_report_is_not_a_download_command():
    """A statement that something was already downloaded cannot be an order.
    Verb lemmatisation must not turn it into one.
    """
    from resolver import is_download_request

    for text in (
        "скачал серию Симпсонов",
        "загрузил третий сезон Симпсонов",
        "скачивал фильм Интерстеллар",
    ):
        assert is_download_request(text) is False, text
        assert parse_request(text) is None, text


def test_an_imperfective_download_verb_is_still_a_download():
    assert parse_request("скачивает третий сезон Симпсонов") == DownloadRequest(
        query="Симпсонов", kind="series", season=3
    )


def test_a_negated_download_is_not_a_download_request():
    """«не скачай/скачивай серию» must stay off the download route even
    after verb forms are normalised to infinitives.
    """
    from resolver import is_download_request

    for text in ("не скачай серию", "не скачивай серию", "не загружай серию"):
        assert is_download_request(text) is False, text
        assert parse_request(text) is None, text


def test_a_question_is_not_a_download():
    assert parse_request("как скачать восьмой сезон?") is None


def test_a_device_command_is_not_a_download():
    assert parse_request("выключи свет в гостиной") is None


def test_a_film_has_no_season():
    assert parse_request("скачай фильм Интерстеллар") == DownloadRequest(
        query="Интерстеллар", kind="movie", season=None
    )


def test_confirmation_uses_the_requested_name():
    assert confirmation_sentence(
        DownloadRequest("Симпсонов", "series", 8)
    ) == "Скачивается восьмой сезон Симпсонов."
    assert confirmation_sentence(
        DownloadRequest("Интерстеллар", "movie", None)
    ) == "Скачивается Интерстеллар."


def _sse(rpc):
    """The MCP server answers streamable HTTP with an SSE `data:` line."""
    return "event: message\ndata: " + json.dumps(rpc) + "\n\n"


def _tool_result(body, *, is_error=False):
    return {
        "jsonrpc": "2.0", "id": 2,
        "result": {
            "content": [{"type": "text", "text": json.dumps(body)}],
            "isError": is_error,
        },
    }


def _mcp_post(result=_tool_result({"success": True, "torrent": {"title": "x"}})):
    """A recording fake of the real MCP HTTP boundary (SSE + session header)."""
    calls = []

    async def post(url, payload, headers=None):
        calls.append((url, payload, dict(headers or {})))
        method = payload.get("method")
        if method == "initialize":
            return (200, _sse({"jsonrpc": "2.0", "id": payload["id"],
                               "result": {"serverInfo": {"name": "torrent-updater"}}}),
                    {"Mcp-Session-Id": "sess-1"})
        if method == "notifications/initialized":
            return (202, "", {})
        if method == "tools/call":
            return (200, _sse(result), {})
        raise AssertionError(f"unexpected MCP method {method!r}")

    return post, calls


@pytest.mark.asyncio
async def test_search_calls_the_mcp_tool_with_the_documented_contract():
    """The MCP server exposes search_and_download(query, media_type, season,
    verify), not the raw HTTP `type` field, and every later call must carry
    the session id that initialize returned."""
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "Симпсоны / Сезон 8"}})
    )
    result = await search_and_download(
        DownloadRequest("Симпсонов", "series", 8),
        url="http://torrent-updater:8020/mcp", post=post,
    )
    assert result["torrent"]["title"] == "Симпсоны / Сезон 8"
    assert [p["method"] for _, p, _ in calls] == [
        "initialize", "notifications/initialized", "tools/call"
    ], calls
    assert all(u == "http://torrent-updater:8020/mcp" for u, _, _ in calls)
    assert calls[1][2].get("Mcp-Session-Id") == "sess-1", calls
    assert calls[2][2].get("Mcp-Session-Id") == "sess-1", calls
    assert calls[2][1]["params"] == {
        "name": "search_and_download",
        "arguments": {
            "query": "Симпсоны", "media_type": "series",
            "season": 8, "verify": False,
        },
    }, calls


@pytest.mark.asyncio
async def test_a_genitive_plural_title_is_sent_in_nominative_plural():
    """Field case 28.11.2026 11:18 MSK: Whisper wrote «Симпсонов» and the
    tool found nothing. A general declension rule must send «Симпсоны» —
    the plural lemma, not the singular one. Only the query SENT to the tool
    changes — the spoken confirmation keeps the user's words.
    """
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "x"}})
    )
    assert await search_and_download(
        DownloadRequest("Симпсонов", "series", 3), url="http://x", post=post,
    )
    assert calls[2][1]["params"]["arguments"]["query"] == "Симпсоны", calls


@pytest.mark.asyncio
async def test_a_declined_noun_is_sent_in_nominative_singular():
    """«серию» is an accusative form; the tracker indexes «серия»."""
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "x"}})
    )
    assert await search_and_download(
        DownloadRequest("серию", "series", 3), url="http://x", post=post,
    )
    assert calls[2][1]["params"]["arguments"]["query"] == "серия", calls


@pytest.mark.asyncio
async def test_an_adjective_agrees_with_its_noun_after_normalisation():
    """«Черного зеркала» must become «Чёрное зеркало», not «Чёрный зеркало»:
    the title filter matches whole words, so a gender mismatch still finds
    nothing even when every word is in nominative form.
    """
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "x"}})
    )
    assert await search_and_download(
        DownloadRequest("Черного зеркала", "series", 3), url="http://x", post=post,
    )
    assert calls[2][1]["params"]["arguments"]["query"] == "Чёрное зеркало", calls


@pytest.mark.asyncio
async def test_a_capitalised_name_is_not_lemmatised_into_a_different_name():
    """pymorphy3 reads «Влада» as a genitive of «Влад». Changing the STEM of
    a capitalised proper noun would make RuTracker search for a different
    title; only ending changes are safe to normalise.
    """
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "x"}})
    )
    assert await search_and_download(
        DownloadRequest("Влада", "series", None), url="http://x", post=post,
    )
    assert calls[2][1]["params"]["arguments"]["query"] == "Влада", calls


@pytest.mark.asyncio
async def test_an_unanalysed_title_is_sent_unchanged():
    """A form the analyser does not recognise must not be mangled."""
    from download import search_and_download

    post, calls = _mcp_post(
        _tool_result({"success": True, "torrent": {"title": "x"}})
    )
    assert await search_and_download(
        DownloadRequest("Interstellar", "movie", None), url="http://x", post=post,
    )
    assert calls[2][1]["params"]["arguments"]["query"] == "Interstellar", calls


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    _tool_result({"success": False, "error": "No results found"}),
    _tool_result({"torrent": {"title": "x"}}),
    _tool_result({"success": True, "torrent": {}}, is_error=True),
    {"jsonrpc": "2.0", "id": 2, "error": {"code": -32603, "message": "boom"}},
])
async def test_a_definite_tool_refusal_is_not_a_confirmation(result):
    from download import search_and_download

    post, calls = _mcp_post(result)
    assert await search_and_download(
        DownloadRequest("Симпсонов", "series", 8),
        url="http://x", post=post,
    ) is None
    assert [p["method"] for _, p, _ in calls][-1] == "tools/call", calls


@pytest.mark.asyncio
async def test_a_timeout_after_the_tool_call_is_unknown_not_a_retry():
    """A timeout after tools/call was sent may still have added the torrent;
    Hermes must not be allowed to add it a second time."""
    from download import UNKNOWN, search_and_download

    async def post(url, payload, headers):
        if payload["method"] == "initialize":
            return (200, _sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
                    {"Mcp-Session-Id": "sess-1"})
        if payload["method"] == "notifications/initialized":
            return (202, "", {})
        raise asyncio.TimeoutError()

    assert await search_and_download(
        DownloadRequest("Симпсонов", "series", 8),
        url="http://x", post=post,
    ) is UNKNOWN


@pytest.mark.asyncio
async def test_a_failure_before_the_tool_call_is_a_definite_failure():
    """Nothing was submitted, so Hermes may safely retry the request."""
    from download import search_and_download

    calls = []

    async def post(url, payload, headers=None):
        calls.append(payload)
        raise RuntimeError("connection refused")

    assert await search_and_download(
        DownloadRequest("Симпсонов", "series", 8),
        url="http://x", post=post,
    ) is None
    assert calls and calls[0]["method"] == "initialize", calls


def _decision(route, conf, reason):
    return type("D", (), {"route": route, "confidence": conf, "reason": reason})()


async def _drive(router, text, stream="livingroom"):
    return [
        json.loads(c.removeprefix("data: "))
        async for c in router._handle(router.RouteRequest(
            text=text, session_id="probe", stream_name=stream
        ))
    ]


@pytest.mark.asyncio
async def test_a_download_turn_acknowledges_then_confirms(monkeypatch):
    import app as router

    requests, chat_calls = [], []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _search(req, *, url=None, post=None):
        requests.append(req)
        return {"success": True, "torrent": {"title": "Симпсоны / Сезон 8"}}

    async def _chat(*a, **k):
        chat_calls.append(1)
        yield "Ответ Гермеса.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.download, "search_and_download", _search)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "скачай восьмой сезон Симпсонов")
    routes = [e for e in events if e.get("type") == "route"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert routes[0]["reason"] == "download_request", routes
    assert requests == [DownloadRequest("Симпсонов", "series", 8)], requests
    assert not chat_calls, "Hermes ran despite the deterministic client succeeding"
    assert sentences == [
        "Пытаюсь скачать.", "Скачивается восьмой сезон Симпсонов."
    ], sentences


@pytest.mark.asyncio
async def test_the_ack_arrives_while_the_browser_search_is_still_running(monkeypatch):
    """28.11.2026 12:29-12:32 MSK: the search took 2 min 24 s and the room
    heard nothing until the torrent was added. The acknowledgement must be
    on the wire BEFORE the tool finishes, not bundled with its result.
    """
    import app as router

    gate = asyncio.Event()
    requests = []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _search(req, *, url=None, post=None):
        requests.append(req)
        await gate.wait()
        return {"success": True, "torrent": {"title": "Симпсоны / Сезон 3"}}

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.download, "search_and_download", _search)

    events = []

    async def collect():
        async for chunk in router._handle(router.RouteRequest(
            text="скачай третий сезон Симпсонов",
            session_id="probe", stream_name="camera-corridor1",
        )):
            events.append(json.loads(chunk.removeprefix("data: ")))

    task = asyncio.create_task(collect())
    for _ in range(50):
        if any(e.get("type") == "sentence" for e in events):
            break
        await asyncio.sleep(0.02)
    early = [e["text"] for e in events if e.get("type") == "sentence"]
    assert requests, "the tool was never called"
    assert not gate.is_set(), "the tool finished before the ack was observed"
    assert early == ["Пытаюсь скачать."], events

    gate.set()
    await asyncio.wait_for(task, 5)
    sentences = [e["text"] for e in events if e.get("type") == "sentence"]
    assert sentences == [
        "Пытаюсь скачать.", "Скачивается третий сезон Симпсонов."
    ], sentences


@pytest.mark.asyncio
async def test_a_definite_download_refusal_is_terse_and_never_calls_hermes(monkeypatch):
    """A parseable request stays deterministic even when the tool refuses it:
    one factual sentence, no tool-agent monologue that guesses a title."""
    import app as router

    chat_calls = []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _search(req, *, url=None, post=None):
        return None

    async def _chat(*a, **k):
        chat_calls.append(k)
        yield "Возможно, вы имели в виду Симпсонов?", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.download, "search_and_download", _search)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "скачай восьмой сезон Симпсонов")
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert not chat_calls, "a refused download must not be guessed by Hermes"
    assert sentences == ["Пытаюсь скачать.", "Не нашла подходящего релиза."], sentences
    assert router._STT_SUSPECT_ACK not in sentences, sentences


def test_gateway_total_budget_covers_a_browser_download():
    """A torrent search measured ~205 s in the field; the old 200 s cap
    would cut the turn just before the confirmation was emitted."""
    from backends import CascadeBackend

    default = inspect.signature(CascadeBackend.__init__).parameters[
        "total_timeout"
    ].default
    assert default == 300, default


@pytest.mark.asyncio
async def test_a_slow_direct_download_keeps_the_sse_socket_alive(monkeypatch):
    """A browser add takes ~205 s. Bytes must reach the gateway before its
    60 s sock_read deadline, or it speaks an apology while the add succeeds."""
    import app as router

    monkeypatch.setattr(router, "EXPERT_KEEPALIVE_S", 0.02)

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _search(req, *, url=None, post=None):
        await asyncio.sleep(0.1)
        return {"success": True, "torrent": {"title": "Симпсоны / Сезон 8"}}

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.download, "search_and_download", _search)

    events = await _drive(router, "скачай восьмой сезон Симпсонов")
    keepalives = [e for e in events if e.get("type") == "keepalive"]
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert len(keepalives) >= 2, events
    assert all(set(e) == {"type"} for e in keepalives), keepalives
    assert sentences == ["Пытаюсь скачать.", "Скачивается восьмой сезон Симпсонов."], sentences


@pytest.mark.asyncio
async def test_a_timed_out_search_never_retriggers_hermes(monkeypatch):
    """A timeout may happen AFTER the torrent was added. Sending the same
    text to Hermes would download it twice — answer honestly instead."""
    import app as router
    from download import UNKNOWN

    chat_calls = []

    async def _classify(*a, **k):
        return _decision("complex_logic", 0.01, "low_confidence(0.01<0.85)")

    async def _search(req, *, url=None, post=None):
        return UNKNOWN

    async def _chat(*a, **k):
        chat_calls.append(1)
        yield "Загрузка запущена.", ""

    monkeypatch.setattr(router.classifier, "classify", _classify)
    monkeypatch.setattr(router.download, "search_and_download", _search)
    monkeypatch.setattr(router.chat_proxy, "stream_chat", _chat)

    events = await _drive(router, "скачай восьмой сезон Симпсонов")
    sentences = [e.get("text") for e in events if e.get("type") == "sentence"]

    assert not chat_calls, "a timeout must not start a second download"
    assert sentences == [
        "Пытаюсь скачать.", "Не удалось подтвердить добавление в закачки."
    ], sentences


@pytest.mark.asyncio
async def test_a_malformed_tool_response_is_unknown_not_a_definite_failure():
    """The call reached the service; an unreadable SSE body cannot be treated
    as a refusal, because the torrent may already be in the queue."""
    from download import UNKNOWN, search_and_download

    post, _ = _mcp_post()
    original = post

    async def broken(url, payload, headers):
        status, text, hdrs = await original(url, payload, headers)
        if payload["method"] == "tools/call":
            return (status, "not an SSE envelope", hdrs)
        return (status, text, hdrs)

    assert await search_and_download(
        DownloadRequest("Симпсонов", "series", 8),
        url="http://x", post=broken,
    ) is UNKNOWN
