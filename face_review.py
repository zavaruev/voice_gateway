"""Face review loop — the Train tab of this house.

Frigate and Double Take both put a human in the loop: a low-scoring but clear
crop can be assigned to a person, and that is how the training set improves.
Our old pipeline had only a CLI enrollment command, so a wrong greeting was
observed in Telegram and then forgotten.

This module writes a SHORT pending record per photo and puts `rv:<id>:<choice>`
into Telegram's inline keyboard. A callback is validated against that record —
callback_data is attacker-controlled text from a chat, so a name from it is
never used as a filesystem path until it is checked against the enrolled
names. Nothing here imports `main`; the watcher owns the photo, this owns the
answer to it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time

import aiohttp

logger = logging.getLogger("face_review")

_RV_RE = re.compile(r"^rv:([0-9a-f]{12}):(.+)$")
_CHOICE_RE = re.compile(r"^[a-zA-Zа-яА-ЯёЁ0-9_-]{1,40}$")


def make_callback(record_id: str, choice: str) -> str:
    return f"rv:{record_id}:{choice}"


def parse_callback(data: str) -> tuple[str, str] | None:
    m = _RV_RE.match((data or "").strip())
    if not m:
        return None
    choice = m.group(2).strip()
    if not _CHOICE_RE.match(choice):
        return None
    return m.group(1), choice


def save_record(review_dir: str, record_id: str, frame_path: str,
                names: list[str], event: str) -> None:
    os.makedirs(review_dir, exist_ok=True)
    rec = {"frame": frame_path, "names": list(names), "event": event,
           "ts": time.time()}
    with open(os.path.join(review_dir, f"{record_id}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)


def load_record(review_dir: str, record_id: str) -> dict | None:
    if not _RV_RE.fullmatch(f"rv:{record_id}:x"):
        return None
    try:
        with open(os.path.join(review_dir, f"{record_id}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def apply_choice(review_dir: str, pending_dir: str, data: str,
                 known: set[str] | None = None) -> str:
    """Validate + copy the reviewed frame. Returns a human-readable verdict.

    `unknown` is not a negative training label (Frigate explicitly refuses
    that): the frame is simply not added to anybody's set.
    """
    parsed = parse_callback(data)
    if not parsed:
        return "ignored: malformed callback"
    rid, choice = parsed
    rec = load_record(review_dir, rid)
    if not rec:
        return "ignored: unknown or expired review record"
    allowed = set(rec.get("names") or []) | {"unknown", "skip"}
    if known is not None:
        allowed |= set(known)
    if choice not in allowed:
        return f"ignored: {choice!r} is not a person in this event"
    frame = rec.get("frame") or ""
    if not os.path.isfile(frame):
        return "ignored: the reviewed frame is gone"
    if choice not in ("unknown", "skip"):
        dst_dir = os.path.join(pending_dir, "_review", choice)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(frame))
        if not os.path.exists(dst):
            shutil.copy2(frame, dst)
    try:
        os.unlink(os.path.join(review_dir, f"{rid}.json"))
    except OSError:
        pass
    return f"reviewed as {choice}: {os.path.basename(frame)}"


async def answer_callback(token: str, callback_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
        async with sess.post(url, data={"callback_query_id": callback_id,
                                        "text": text[:180]}) as resp:
            await resp.read()


async def review_loop(token: str, review_dir: str, pending_dir: str,
                      known: set[str], stop: asyncio.Event | None = None) -> None:
    """Long-poll the FACE bot (not the assistant's token) for inline answers."""
    offset = 0
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    while stop is None or not stop.is_set():
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as sess:
                async with sess.get(url, params={"offset": offset, "timeout": 30}) as resp:
                    data = await resp.json()
            for update in data.get("result") or []:
                offset = max(offset, int(update.get("update_id", 0)) + 1)
                cb = update.get("callback_query")
                if not cb:
                    continue
                answer = apply_choice(
                    review_dir, pending_dir, cb.get("data") or "", known)
                logger.info(f"[REVIEW] {answer}")
                await answer_callback(token, cb.get("id", ""), answer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            logger.warning(f"[REVIEW] poll failed: {exc!r}")
            await asyncio.sleep(5)
