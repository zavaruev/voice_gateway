"""Kitchen scale MCP client and deterministic spoken verdicts (step 1).

The ESP32 scale at 192.168.22.123 serves a stateless JSON-RPC 2.0 endpoint at
POST /mcp. This module is deliberately free of FastAPI/HA imports so it can be
unit-tested offline and later reused by the worker for recipe-guided cooking.

Two commands are supported: read the weight, and tare. The tare tool is the
firmware's Zero/0 button (weight_task_tare -> scale_filter_tare(hard=true)):
it stores the CURRENT reading as the new zero, even with a load on the platform.
Every failure becomes an honest Russian sentence — nothing here ever guesses a
reading or claims a tare it did not verify.
"""

import json
import logging
import math

import aiohttp

from config import SCALE_MCP_URL
from resolver import _cardinal

logger = logging.getLogger("scale")

_TIMEOUT_S = 3.0
_RPC_ID = 1

_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста",
    "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот",
)


async def _default_post(url: str, payload: dict) -> dict:
    """POST one JSON-RPC message and return the parsed JSON body."""
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"scale HTTP {resp.status}")
            return await resp.json(content_type=None)


async def _rpc(method: str, params: dict, *, url=None, post=None):
    """One JSON-RPC call; raises on transport, protocol or tool error."""
    payload = {
        "jsonrpc": "2.0", "id": _RPC_ID, "method": method, "params": params,
    }
    data = await (post or _default_post)(url or SCALE_MCP_URL, payload)
    if not isinstance(data, dict):
        raise ValueError("scale returned a non-object reply")
    if "error" in data:
        raise ValueError(str(data["error"]))
    result = data.get("result")
    if isinstance(result, dict) and result.get("isError"):
        raise ValueError("scale tool reported isError")
    return result


async def get_weight(*, url=None, post=None) -> dict | None:
    """A fresh reading, or None on any failure (never a stale/fabricated one)."""
    try:
        result = await _rpc(
            "tools/call", {"name": "get_weight", "arguments": {}},
            url=url, post=post,
        )
        content = result.get("content") if isinstance(result, dict) else None
        text = content[0].get("text") if content and isinstance(content, list) else None
        reading = json.loads(text) if isinstance(text, str) else None
        if not isinstance(reading, dict):
            raise ValueError("get_weight returned no reading")
        float(reading["grams"])  # validate the number we are about to speak
        return reading
    except Exception as e:
        logger.warning("scale get_weight failed: %r", e)
        return None


async def tare(*, url=None, post=None) -> bool:
    """Call the tare tool; True only on a clean, non-error reply."""
    try:
        await _rpc(
            "tools/call", {"name": "tare", "arguments": {}},
            url=url, post=post,
        )
        return True
    except Exception as e:
        logger.warning("scale tare failed: %r", e)
        return False


async def tare_checked(*, url=None, post=None) -> tuple[bool, str]:
    """Tare (Zero-button semantics) then verify. Returns (success, reason)."""
    before = await get_weight(url=url, post=post)
    if before is None:
        return False, "unreachable"
    logger.info("scale tare: current reading %s g becomes the new zero",
                before.get("grams"))
    if not await tare(url=url, post=post):
        return False, "failed"
    after = await get_weight(url=url, post=post)
    if after is None or not after.get("zero"):
        logger.warning("scale tare not verified by a fresh zero reading")
        return False, "unverified"
    return True, "ok"


def _spelled(n: int) -> str:
    """0..999 in words; above that leave digits for the TTS num2ru step."""
    if n >= 1000:
        return str(n)
    h, rest = divmod(n, 100)
    parts = [_HUNDREDS[h]] if h else []
    # res._cardinal(0) is "ноль" — correct for 0 itself, wrong for 100
    # ("сто ноль"). Only spell the tail when it exists or the whole number is 0.
    if rest or not h:
        parts.append(_cardinal(rest))
    return " ".join(parts)


def _gram_noun(n: int) -> str:
    last2, last = n % 100, n % 10
    if 11 <= last2 <= 14:
        return "граммов"
    if last == 1:
        return "грамм"
    if 2 <= last <= 4:
        return "грамма"
    return "граммов"


def weight_sentence(weight: dict | None) -> str | None:
    """One speakable sentence, whole grams, correct noun agreement."""
    if not isinstance(weight, dict):
        return None
    try:
        grams = float(weight["grams"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(grams):
        return None
    n = int(grams + 0.5) if grams >= 0 else -int(-grams + 0.5)
    prefix = "минус " if n < 0 else ""
    return f"На весах {prefix}{_spelled(abs(n))} {_gram_noun(abs(n))}."


def sentence_for(kind: str, *, reason: str = "", weight: dict | None = None) -> str:
    """The one sentence the caller will speak, for any outcome."""
    if kind == "weight":
        if reason == "unreachable":
            return "Весы не отвечают."
        return weight_sentence(weight) or "Не удалось прочитать вес."
    return {
        "ok": "Обнулила.",
        "unreachable": "Весы не отвечают.",
        "failed": "Не получилось обнулить весы.",
        "unverified": "Не получилось обнулить весы.",
    }.get(reason, "Не получилось обнулить весы.")


async def handle(kind: str, *, url=None, post=None) -> str:
    """Entry point used by app._handle — always returns speakable text."""
    if kind == "weight":
        reading = await get_weight(url=url, post=post)
        return sentence_for(
            "weight", reason="unreachable" if reading is None else "", weight=reading
        )
    ok, reason = await tare_checked(url=url, post=post)
    return sentence_for("tare", reason="ok" if ok else reason)
