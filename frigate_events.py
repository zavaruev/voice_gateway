"""Inject a Frigate 'person' snapshot into an active face event.

Measured pain: three arrivals on 18.11 produced `face_frames: 0` in 40
attempts — the HA door+motion trigger fires before anybody is in the camera's
frame, and the burst then spends 45 s photographing an empty hallway.
Frigate already detects `person` continuously and its event API carries the
snapshot of the moment it saw one (the same source Double Take consumes via
MQTT). We use its HTTP API so the gateway needs no new dependency.

This is a SUPPLEMENT to the existing capture path, never a replacement: a
snapshot is added only to an event that is already being collected, so there
is no second greeting and no second Telegram notification. No person event,
no injection — the old behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import aiohttp

logger = logging.getLogger("frigate_events")

FRIGATE_URL = os.getenv("FRIGATE_URL", "http://192.168.22.102:5000").rstrip("/")
FACE_API = os.getenv("FACE_API_URL", "http://192.168.22.102:8090").rstrip("/")
CAPTURES_DIR = os.getenv("FACE_CAPTURES_DIR",
                         "/mnt/media/docker-compose/face-captures/captures")
POLL_S = float(os.getenv("FRIGATE_POLL_S", "2"))
MAX_AGE_S = float(os.getenv("FRIGATE_EVENT_MAX_AGE_S", "20"))
ACTIVE_S = float(os.getenv("FRIGATE_ACTIVE_EVENT_S", "60"))
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EVENT_RE = re.compile(r"^(\d{2}-\d{2}-\d{2})_")


def active_event(captures_dir: str, now: float) -> tuple[str, str] | None:
    """Newest face event with a file touched in the last ACTIVE_S seconds."""
    try:
        days = sorted((d for d in os.listdir(captures_dir)
                       if DAY_RE.match(d)), reverse=True)
    except OSError:
        return None
    for day in days:
        day_dir = os.path.join(captures_dir, day)
        newest: dict[str, float] = {}
        try:
            names = os.listdir(day_dir)
        except OSError:
            continue
        for name in names:
            m = EVENT_RE.match(name)
            if not m or "_frigate." in name:
                continue
            try:
                mt = os.path.getmtime(os.path.join(day_dir, name))
            except OSError:
                continue
            prefix = m.group(1)
            if mt > newest.get(prefix, 0.0):
                newest[prefix] = mt
        fresh = [(mt, p) for p, mt in newest.items() if now - mt <= ACTIVE_S]
        if fresh:
            mt, prefix = max(fresh)
            return day, prefix
    return None


class FrigateEventInjector:
    def __init__(self):
        self.seen: set[str] = set()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="frigate-events")
            logger.info(f"face events: Frigate person snapshots -> {CAPTURES_DIR}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _detect(self, session, image: bytes) -> dict:
        data = aiohttp.FormData()
        data.add_field("file", image, filename="frigate.jpg",
                       content_type="image/jpeg")
        async with session.post(f"{FACE_API}/detect", data=data,
                                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return {}
            body = await resp.json()
        faces = body.get("faces") or []
        if not faces:
            return {}
        best = max(faces, key=lambda f: int(f.get("face_size") or 0))
        return {"quality_ok": True,
                "face_size": int(best.get("face_size") or 0),
                "det_score": float(best.get("det_score") or 0.0)}

    async def poll_once(self, session) -> bool:
        """Returns True when a frame was injected."""
        async with session.get(f"{FRIGATE_URL}/api/events",
                               params={"limit": 10, "type": "person"},
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return False
            events = await resp.json()
        now = time.time()
        for ev in events:
            eid = ev.get("id")
            if not eid or eid in self.seen:
                continue
            self.seen.add(eid)
            if (not ev.get("has_snapshot")
                    or now - float(ev.get("start_time") or 0) > MAX_AGE_S):
                continue
            active = active_event(CAPTURES_DIR, now)
            if not active:
                continue
            day, prefix = active
            cam = ev.get("camera") or "entrance1"
            day_dir = os.path.join(CAPTURES_DIR, day)
            path = os.path.join(day_dir, f"{prefix}_camera_{cam}_face90_frigate.jpg")
            if os.path.exists(path):
                continue
            async with session.get(
                    f"{FRIGATE_URL}/api/events/{eid}/snapshot.jpg",
                    timeout=aiohttp.ClientTimeout(total=10)) as snap:
                if snap.status != 200:
                    continue
                image = await snap.read()
            verdict = await self._detect(session, image)
            if not verdict.get("quality_ok"):
                continue
            with open(path, "wb") as f:
                f.write(image)
            verdict["path"] = path
            with open(os.path.join(day_dir, f"{prefix}_frames.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(verdict) + "\n")
            meta_path = os.path.join(day_dir, f"{prefix}_meta.json")
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
            meta.setdefault("frame_details", []).append(verdict)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            logger.info(f"[FRIGATE] {eid} ({cam}) -> {day}/{prefix}: "
                        f"face={verdict['face_size']}px "
                        f"det={verdict['det_score']:.3f}")
            return True
        return False

    async def run(self) -> None:
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    await self.poll_once(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                                # noqa: BLE001
                logger.warning(f"[FRIGATE] poll failed: {exc!r}")
            await asyncio.sleep(POLL_S)
