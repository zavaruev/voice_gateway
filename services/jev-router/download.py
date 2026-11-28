"""Direct download requests — parse the spoken phrase and call torrent-updater.

27.11.2026. Hermes can find and add a torrent, but its tool loop took 3 min
25 s on «скачай восьмой сезон Симпсонов» while the voice turn had already
been cut. The service that ADDS the torrent already returns a structured
success with the title, so a deterministic request goes straight there and
the router speaks «Скачивается <name>» at that moment. No LLM, no completion
announcement. A phrase this narrow parser cannot read keeps the Hermes route.

27.11.2026 (second pass): torrent-updater's documented interface is its MCP
server on :8020/mcp, not the raw REST endpoint. We call its
`search_and_download` tool over streamable HTTP: initialize -> notification
-> tools/call, carrying the session id and reading the result from the SSE
`data:` line. The tool body is a JSON string in `result.content[0].text`.

The `post` injection follows `scale.py`: unit tests never touch the network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from resolver import is_download_request, normalize_stt_verbs
from query_norm import normal_query, normalise_download_verb

logger = logging.getLogger("router.download")

# The HTTP call can time out AFTER torrent-updater added the torrent. That
# outcome is not "failure": retrying through Hermes would add it a second
# time. The router says it could not confirm, and stops.
UNKNOWN = object()

# Whole-token ordinals for seasons. Written out, not generated: Whisper says
# «восьмой», not «8-й», and a prefix match would confuse «восьмой» with
# «восемнадцатый». Cyrillic «ё» is optional on input.
_ORDINALS: dict[str, int] = {
    "первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5,
    "шестой": 6, "седьмой": 7, "восьмой": 8, "девятый": 9, "десятый": 10,
    "одиннадцатый": 11, "двенадцатый": 12, "тринадцатый": 13,
    "четырнадцатый": 14, "пятнадцатый": 15, "шестнадцатый": 16,
    "семнадцатый": 17, "восемнадцатый": 18, "девятнадцатый": 19,
    "двадцатый": 20,
}
_ORDINALS_REV = {v: k for k, v in _ORDINALS.items()}
_WAKE_WORDS = ("компьютер", "домовой", "добби")
_VERB_RE = re.compile(
    r"^(скачай|скачать|скачивать|скачайте|загрузи|загрузить|загружать|"
    r"загрузите|качай|качать)\b",
    re.IGNORECASE,
)
_SERIES_RE = re.compile(
    r"\b(сезон\w*|сериал\w*|сери[яию]\w*|эпизод\w*)\b", re.IGNORECASE
)
_MOVIE_RE = re.compile(r"\b(фильм\w*|кино|мультфильм\w*)\b", re.IGNORECASE)
_SEASON_RE = re.compile(
    r"\b(\d{1,2}|[а-яё]+)\s+(?:сезон\w*|сери[яию]\w*|эпизод\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DownloadRequest:
    query: str
    kind: str
    season: int | None


def _ordinal(token: str) -> int | None:
    token = token.replace("ё", "е")
    if token.isdigit():
        return int(token)
    return _ORDINALS.get(token)


def parse_request(text: str) -> DownloadRequest | None:
    """«скачай восьмой сезон Симпсонов» -> query/kind/season, else None."""
    t = normalize_stt_verbs(text or "").strip()
    if not is_download_request(t):
        return None
    tokens = t.split()
    if tokens and tokens[0].lower() in _WAKE_WORDS:
        tokens = tokens[1:]
        t = " ".join(tokens)
    t = normalise_download_verb(t)
    verb = _VERB_RE.match(t)
    if not verb:
        return None
    rest = t[verb.end():].strip(" ,.!?:;-")

    season: int | None = None
    sm = _SEASON_RE.search(rest)
    if sm:
        season = _ordinal(sm.group(1))
        if season is None:
            return None
        rest = (rest[:sm.start()] + " " + rest[sm.end():]).strip()

    series_word = bool(_SERIES_RE.search(t))
    rest = _SERIES_RE.sub(" ", rest)
    rest = _MOVIE_RE.sub(" ", rest)
    rest = re.sub(r"\s+", " ", rest).strip(" ,.!?:;-")
    if len(rest) < 2:
        return None
    return DownloadRequest(rest, "series" if series_word else "movie", season)


def confirmation_sentence(req: DownloadRequest) -> str:
    """Name what was requested — the user's own words identify it best."""
    if req.season:
        ordinal = _ORDINALS_REV.get(req.season, str(req.season))
        return f"Скачивается {ordinal} сезон {req.query}."
    return f"Скачивается {req.query}."


class MCPToolRefused(Exception):
    """The MCP server answered with a definite tool/protocol error."""


def _search_query(query: str) -> str:
    """Lemmatise the OUTBOUND query so any Russian declension is searchable.

    RuTracker indexes normal forms; a declined title returns only the site's
    search-help row. confirmation_sentence still speaks the user's own words.
    """
    return normal_query(query)


_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}
_INIT_PARAMS = {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "jev-router", "version": "1"},
}


def _rpc(method: str, params: dict | None = None, rpc_id: int | None = None) -> dict:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if rpc_id is not None:
        msg["id"] = rpc_id
    return msg


def _parse_rpc(text: str, rpc_id: int) -> dict:
    """Read one JSON-RPC result from a streamable-HTTP body (JSON or SSE)."""
    if not text or not text.strip():
        raise ValueError("empty MCP response")
    try:
        msg = json.loads(text)
    except ValueError:
        msg = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip())
                break
        if msg is None:
            raise ValueError("no SSE data in MCP response")
    if msg.get("id") != rpc_id:
        raise ValueError(f"unexpected MCP id: {msg.get('id')!r}")
    if "error" in msg:
        raise MCPToolRefused(msg["error"])
    if "result" not in msg:
        raise ValueError("MCP response has no result")
    return msg["result"]


def _tool_body(result: dict) -> dict:
    """FastMCP serialises a tool's dict as `content[0].text` (a JSON string)."""
    if not isinstance(result, dict):
        raise ValueError("MCP result is not an object")
    if result.get("isError"):
        raise MCPToolRefused(result.get("content"))
    content = result.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("MCP tool result has no content")
    body = json.loads(content[0].get("text") or "")
    if not isinstance(body, dict):
        raise ValueError("MCP tool body is not an object")
    return body


async def _default_post(url: str, payload: dict, headers: dict) -> tuple[int, str, dict]:
    import aiohttp

    from config import DOWNLOAD_TIMEOUT_S

    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_S, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            return resp.status, await resp.text(), dict(resp.headers)


async def search_and_download(
    req: DownloadRequest, *, url: str | None = None, post=None
) -> dict | None:
    """Call the MCP tool that adds the torrent; None means no confirmation.

    Only `success is True` in the TOOL BODY counts — MCP answers 200 even on
    a refused tool. Nothing is retried: the browser search is expensive, and
    a timeout after tools/call was sent may mean the torrent was already
    added (caller must not hand the text to Hermes in that case).
    """
    from config import TORRENT_UPDATER_MCP_URL

    target = (url or TORRENT_UPDATER_MCP_URL).rstrip("/")
    do_post = post or _default_post
    sent = False
    try:
        _, text, hdrs = await do_post(
            target, _rpc("initialize", _INIT_PARAMS, 1), dict(_MCP_HEADERS)
        )
        _parse_rpc(text, 1)
        session_id = hdrs.get("Mcp-Session-Id") or hdrs.get("mcp-session-id")
        if not session_id:
            raise ValueError("MCP initialize returned no session id")
        headers = dict(_MCP_HEADERS)
        headers["Mcp-Session-Id"] = session_id
        await do_post(target, _rpc("notifications/initialized"), headers)
        arguments = {
            "query": _search_query(req.query),
            "media_type": req.kind,
            "season": req.season,
            "verify": False,
        }
        sent = True
        _, text, _ = await do_post(
            target,
            _rpc("tools/call", {
                "name": "search_and_download",
                "arguments": arguments,
            }, 2),
            headers,
        )
        body = _tool_body(_parse_rpc(text, 2))
    except MCPToolRefused as e:
        logger.warning("search-and-download tool refused: %s", e)
        return None
    except (asyncio.TimeoutError, TimeoutError, ValueError) as e:
        if sent:
            logger.warning("search-and-download outcome unknown (add may have happened): %s", e)
            return UNKNOWN
        logger.warning("search-and-download failed before submission: %s", e)
        return None
    except Exception as e:
        if sent:
            logger.warning("search-and-download outcome unknown (add may have happened): %s", e)
            return UNKNOWN
        logger.warning("search-and-download failed: %s", e)
        return None
    if body.get("success") is not True:
        logger.warning(
            "search-and-download refused: %s", body.get("error") or repr(body)
        )
        return None
    torrent = body.get("torrent")
    logger.info(
        "torrent added: %s",
        torrent.get("title", "") if isinstance(torrent, dict) else repr(torrent),
    )
    return body
