"""Face-arrival event watcher — stage 2 of the greeting migration.

The old `face-greeting-handler` container owned event discovery, streaming
recognition, voting, Telegram and enrollment saving, AND had its own voice
stack. This module moves ALL of that into voice_gateway so there is one log,
one auth model and one playback path (echo ring + arbiter intact).

WHAT IS DELIBERATELY UNCHANGED (port, not redesign)
    - edge trigger events come from the capture daemon's files;
    - streaming recognition: greet as soon as MIN_FRAMES_FOR_MATCH frames agree;
    - the vote rule, VOTE logs, AMBIGUITY_MARGIN, VOTE_FLOOR, marina's
      TTS_SUPPRESS_WHEN_ALONE, cooldown, startup pre-load, per-event try/except;
    - high-confidence frames still go to pending/ for the face-retrain service.

WHAT IS DELIBERATELY GONE
    - edge-tts, ffmpeg limiter, direct /play_audio POST and its busy-guard:
      speech goes through face_greeting.prepare() + voice_mcp._speak_tool();
    - the HTTP /api/face/greeting hop (same process now).

CAPTURES are mounted at the SAME absolute path as on the host: meta.json from
the daemon carries absolute frame paths, and rewriting them would be a second
source of truth. Never import `main` here (the telegram_client cycle rule).
"""

import asyncio
import glob
import html
import json
import logging
import os
import re
import shutil
import time

import aiohttp

import face_greeting
import identity_client
import face_review
import frigate_events
import voice_mcp
logger = logging.getLogger("face_events")

# --- config (all overridable; defaults are the face-captures production ones)
CAPTURES_DIR = os.getenv("FACE_CAPTURES_DIR",
                         "/mnt/media/docker-compose/face-captures/captures")
PENDING_DIR = os.getenv("FACE_PENDING_DIR",
                        "/mnt/media/docker-compose/face-captures/pending")
FACE_API = os.getenv("FACE_API_URL", "http://192.168.22.102:8090").rstrip("/")
FACE_ROOM = os.getenv("FACE_GREETING_ROOM", "entrance1")
TG_TOKEN = os.getenv("FACE_TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("FACE_TG_CHAT_ID", "543867556")

RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.35"))
AMBIGUITY_MARGIN = float(os.getenv("FACE_AMBIGUITY_MARGIN", "0.08"))
MIN_FRAMES_FOR_MATCH = int(os.getenv("FACE_MIN_FRAMES_FOR_MATCH", "2"))
FRAME_VOTER_SLACK = float(os.getenv("FACE_FRAME_VOTER_SLACK", "0.05"))
VOTE_FLOOR = float(os.getenv("FACE_VOTE_FLOOR", "0.20"))
MAX_FRAMES_PER_EVENT = int(os.getenv("FACE_MAX_FRAMES_PER_EVENT", "12"))
META_WAIT_SECONDS = float(os.getenv("FACE_META_WAIT_SECONDS", "60"))
STREAM_POLL_INTERVAL = float(os.getenv("FACE_STREAM_POLL_INTERVAL", "0.5"))
POLL_INTERVAL = float(os.getenv("FACE_POLL_INTERVAL", "2"))
COOLDOWN_SECONDS = int(os.getenv("FACE_COOLDOWN", "30"))
INITIAL_EVENTS_MAX_AGE = int(os.getenv("FACE_INITIAL_EVENTS_MAX_AGE", "120"))
CONTINUAL_ENROLLMENT_THRESHOLD = float(
    os.getenv("FACE_CONTINUAL_ENROLLMENT_THRESHOLD", "0.6"))
# Keep the identity gallery at the best 24 frames. Off in code; enabled in
# compose only after the one-time top-up has been verified.
FACE_BEST_FRAMES_ENABLED = os.getenv("FACE_BEST_FRAMES_ENABLED", "").lower() in ("1", "true", "yes")
FACE_BEST_FRAME_MIN_GAP_S = int(os.getenv("FACE_BEST_FRAME_MIN_GAP_S", "3600"))
_LAST_CURATED = {}
# Nobody is suppressed by default. Marina used to be muted when alone because
# her enrolment was being rebuilt; the user asked to greet her again (20.11.2026).
TTS_SUPPRESS_WHEN_ALONE = {
    n.strip().lower() for n in
    os.getenv("FACE_TTS_SUPPRESS_WHEN_ALONE", "").split(",") if n.strip()
}
FACE_WEIGHTED_VOTE = os.getenv("FACE_WEIGHTED_VOTE", "").lower() in ("1", "true", "yes")
CAMERA_IDS = ("camera_entrance1", "camera_entrance2")

EVENT_PATTERN = re.compile(r"^(\d{2}-\d{2}-\d{2})_")
UNKNOWN_META_PATTERN = re.compile(r"^(\d{2}-\d{2}-\d{2})_unknown_meta\.json$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TG_RETRIES = 3
TG_RETRY_DELAY = 2
# Review loop (Frigate "Train tab" via Telegram inline buttons). Off by
# default: it changes what a door photo looks like in the chat.
FACE_REVIEW_ENABLED = os.getenv("FACE_REVIEW_ENABLED", "").lower() in ("1", "true", "yes")
FRIGATE_EVENTS_ENABLED = os.getenv("FRIGATE_EVENTS_ENABLED", "").lower() in ("1", "true", "yes")
REVIEW_DIR = os.getenv("FACE_REVIEW_DIR", os.path.join(PENDING_DIR, "_review_records"))
KNOWN_NAMES = {n.strip() for n in os.getenv("FACE_KNOWN_NAMES", "").split(",") if n.strip()}


# --- pure helpers (unit-tested without a network or a camera) -------------


def event_key(day_dir: str, event_prefix: str) -> str:
    return f"{os.path.basename(day_dir)}/{event_prefix}"


def date_dirs() -> list:
    if not os.path.isdir(CAPTURES_DIR):
        return []
    dirs = [d for d in os.listdir(CAPTURES_DIR)
            if DATE_PATTERN.match(d) and
            os.path.isdir(os.path.join(CAPTURES_DIR, d))]
    return sorted(dirs, reverse=True)


def existing_events(now: float | None = None) -> set:
    """Events older than INITIAL_EVENTS_MAX_AGE are already handled — a restart
    must not re-send last week's Telegram photos. ANY file identifies an event
    (a burst that died before meta.json is still an event).
    """
    if now is None:
        now = time.time()
    out: set = set()
    for day in date_dirs():
        newest: dict = {}
        for path in glob.glob(os.path.join(CAPTURES_DIR, day, "*")):
            m = EVENT_PATTERN.match(os.path.basename(path))
            if not m:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            prefix = m.group(1)
            if mtime > newest.get(prefix, 0.0):
                newest[prefix] = mtime
        for prefix, mtime in newest.items():
            if now - mtime >= INITIAL_EVENTS_MAX_AGE:
                out.add(event_key(os.path.join(CAPTURES_DIR, day), prefix))
    return out


def find_new_events(day_dir: str, processed: set) -> tuple:
    faces, unknown = set(), set()
    for path in sorted(glob.glob(os.path.join(day_dir, "*_face*.jpg"))):
        m = EVENT_PATTERN.match(os.path.basename(path))
        if m and event_key(day_dir, m.group(1)) not in processed:
            faces.add(m.group(1))
    for path in sorted(glob.glob(os.path.join(day_dir, "*_unknown_meta.json"))):
        m = UNKNOWN_META_PATTERN.match(os.path.basename(path))
        if m and event_key(day_dir, m.group(1)) not in processed:
            unknown.add(m.group(1))
    return faces, unknown


def load_meta(day_dir: str, event_prefix: str) -> dict:
    path = os.path.join(day_dir, f"{event_prefix}_meta.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def event_frames(day_dir: str, event_prefix: str) -> list:
    """Best frames first, using the daemon's own quality verdicts."""
    details = load_meta(day_dir, event_prefix).get("frame_details") or []
    if details:
        ranked = sorted(details, key=lambda d: (bool(d.get("quality_ok")),
                                                int(d.get("face_size") or 0),
                                                float(d.get("det_score") or 0.0)),
                        reverse=True)
        paths = []
        for d in ranked:
            p = d.get("path")
            if p and os.path.exists(p) and p not in paths:
                paths.append(p)
        if paths:
            return paths
    paths = []
    for cam_id in CAMERA_IDS:
        paths.extend(sorted(glob.glob(
            os.path.join(day_dir, f"{event_prefix}_{cam_id}_face*.jpg"))))
    return paths


def frame_verdicts(day_dir: str, event_prefix: str) -> dict:
    """The daemon appends a JSON line per frame as it writes it; an unseen
    verdict means 'wait', NOT 'accept' — the first frames are the blurriest.
    """
    path = os.path.join(day_dir, f"{event_prefix}_frames.jsonl")
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                except ValueError:
                    continue
                out[rec.get("path")] = bool(rec.get("quality_ok"))
    except OSError:
        pass
    return out


def face_areas_from_meta(day_dir: str, event_prefix: str) -> dict:
    """The daemon already publishes face_size per frame in meta.json. Frigate's
    rule — bigger, closer faces count more — needs no new measurement here."""
    out: dict = {}
    for d in load_meta(day_dir, event_prefix).get("frame_details") or []:
        path = d.get("path")
        if path and d.get("quality_ok"):
            try:
                out[path] = int(d.get("face_size") or 0)
            except (TypeError, ValueError):
                out[path] = 0
    return out


def votes_to_recognized(votes: dict, event_prefix: str,
                        face_areas: dict | None = None) -> list:
    """Port of the handler's vote rule, plus Frigate-style area weighting.

    `face_areas` maps frame path -> face size in px. When present the
    per-person confidence is the area-weighted mean over that person's voting
    frames (weight 1..3, a 60 px face still counts), so a large clear face
    outweighs a distant one without ever increasing the FRAME COUNT needed to
    greet. Without it the original max-confidence rule is unchanged.
    """
    recognized = []
    for name, entries in votes.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        best = entries[0]
        if len(entries) < MIN_FRAMES_FOR_MATCH:
            logger.info(f"[VOTE] {event_prefix}: {name} only in {len(entries)} "
                        f"frame(s), best {best[0]:.3f} - needs "
                        f"{MIN_FRAMES_FOR_MATCH}, not greeting")
            continue
        over = sum(1 for c, _, _ in entries if c >= RECOGNITION_THRESHOLD)
        if over < len(entries):
            logger.info(f"[VOTE] {event_prefix}: {name} confirmed by "
                        f"{len(entries)} frames (best {best[0]:.3f}, "
                        f"{over} at/above {RECOGNITION_THRESHOLD}, "
                        f"slack {FRAME_VOTER_SLACK})")
        conf = best[0]
        if face_areas:
            weights = [max(1.0, min(3.0, int(face_areas.get(e[1], 0) or 0) / 200.0))
                       for e in entries]
            conf = sum(e[0] * w for e, w in zip(entries, weights)) / sum(weights)
            logger.info(f"[VOTE] {event_prefix}: {name} area-weighted "
                        f"{conf:.3f} (raw best {best[0]:.3f})")
        recognized.append((name, conf, best[1], best[2]))

    best_by_name = {}
    for name, conf, frame_path, cam_id in recognized:
        if name not in best_by_name or conf > best_by_name[name][1]:
            best_by_name[name] = (name, conf, frame_path, cam_id)
    ranked = sorted(best_by_name.values(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return []
    top = ranked[0][1]
    keep = [ranked[0]]
    for item in ranked[1:]:
        gap = top - item[1]
        if gap >= AMBIGUITY_MARGIN:
            keep.append(item)
        else:
            logger.info(f"[AMBIG] {event_prefix}: dropping {item[0]} "
                        f"{item[1]:.3f} - only {gap:.3f} behind "
                        f"{ranked[0][0]} {top:.3f} (margin {AMBIGUITY_MARGIN})")
    return keep


def who_to_greet(recognized: list) -> set:
    """marina alone is not greeted (her enrolment is still being rebuilt);
    Telegram is still sent. With company she is greeted normally."""
    names = {r[0] for r in recognized}
    if len(names) == 1 and names <= TTS_SUPPRESS_WHEN_ALONE:
        return set()
    return names


# --- the watcher -----------------------------------------------------------


class FaceGreetingWatcher:
    def __init__(self):
        self.processed_events: set = existing_events()
        self.last_trigger = 0.0
        self._task: asyncio.Task | None = None
        self._review_task: asyncio.Task | None = None
        self._frigate = frigate_events.FrigateEventInjector() if FRIGATE_EVENTS_ENABLED else None
        logger.info(f"[INIT] pre-loaded {len(self.processed_events)} existing "
                    f"face events (face_events)")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="face-events")
            logger.info(f"face events: watching {CAPTURES_DIR} -> "
                        f"{FACE_ROOM} via face_greeting")
        if (FACE_REVIEW_ENABLED and TG_TOKEN and self._review_task is None):
            self._review_task = asyncio.create_task(
                face_review.review_loop(TG_TOKEN, REVIEW_DIR, PENDING_DIR,
                                        KNOWN_NAMES), name="face-review")
        if self._frigate is not None:
            self._frigate.start()

    async def stop(self) -> None:
        if self._frigate is not None:
            await self._frigate.stop()
        for task in (self._task, self._review_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._review_task = None

    # --- recognition -------------------------------------------------------
    async def _recognize_frame(self, session, frame_path: str,
                               threshold: float) -> list:
        try:
            with open(frame_path, "rb") as fh:
                image = fh.read()
            ident = identity_client.IDENTITY
            if ident is not None:
                try:
                    return await ident.recognize(image, threshold)
                except Exception as exc:
                    logger.warning(f"[WARN] identity /recognize failed: {exc!r}; "
                                   "using the face API directly")
            data = aiohttp.FormData()
            data.add_field("file", image, filename=os.path.basename(frame_path),
                           content_type="image/jpeg")
            data.add_field("threshold", str(threshold))
            async with session.post(
                    f"{FACE_API}/recognize", data=data,
                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning(f"[WARN] /recognize HTTP {resp.status} "
                                       f"for {os.path.basename(frame_path)}")
                        return []
                    return (await resp.json()).get("faces", [])
        except Exception as exc:
            logger.warning(f"[WARN] recognition failed for {frame_path}: {exc}")
            return []

    async def _recognize_streaming(self, day_dir: str, event_prefix: str) -> list:
        """Greet as soon as the vote passes, keep collecting for pending/."""
        frame_glob = os.path.join(day_dir, f"{event_prefix}_camera_*_face*.jpg")
        meta_path = os.path.join(day_dir, f"{event_prefix}_meta.json")
        deadline = time.monotonic() + META_WAIT_SECONDS
        seen: set = set()
        judged: set = set()
        votes: dict = {}
        greeted = False
        areas = face_areas_from_meta(day_dir, event_prefix) if FACE_WEIGHTED_VOTE else None
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                verdicts = frame_verdicts(day_dir, event_prefix)
                for frame_path in sorted(glob.glob(frame_glob)):
                    if frame_path in judged:
                        continue
                    verdict = verdicts.get(frame_path)
                    if verdict is None:
                        continue          # the daemon has not published it yet
                    judged.add(frame_path)
                    if not verdict:
                        continue
                    if len(seen) >= MAX_FRAMES_PER_EVENT:
                        break
                    seen.add(frame_path)
                    base = os.path.basename(frame_path)
                    cam_id = ("camera_entrance1" if "_camera_entrance1_" in base
                              else "camera_entrance2")
                    for face in await self._recognize_frame(session, frame_path, 0.0):
                        name = face.get("name", "unknown")
                        conf = float(face.get("confidence", 0) or 0)
                        if name == "unknown" or conf < VOTE_FLOOR:
                            continue
                        votes.setdefault(name, []).append((conf, frame_path, cam_id))
                    if greeted:
                        continue
                    candidate = votes_to_recognized(votes, event_prefix, areas)
                    if not candidate:
                        continue
                    greeted = True
                    logger.info(f"[RECOG] {event_prefix}: greeting on "
                                f"{len(seen)} frame(s) seen so far")
                    await self._greet_and_notify(candidate, event_prefix)
                if os.path.exists(meta_path):
                    if len(glob.glob(frame_glob)) <= len(judged):
                        break
                await asyncio.sleep(STREAM_POLL_INTERVAL)
        # meta.json exists by now (or the wait expired) — re-read the areas so
        # the FINAL vote uses every frame's size, even during a live burst.
        if FACE_WEIGHTED_VOTE:
            areas = face_areas_from_meta(day_dir, event_prefix)
        return votes_to_recognized(votes, event_prefix, areas)

    # --- outputs -----------------------------------------------------------
    async def _greet_and_notify(self, recognized: list, event_prefix: str) -> None:
        logger.info(f"[RECOGNIZED] {[(r[0], round(r[1], 3)) for r in recognized]}")
        names = who_to_greet(recognized)
        text = ""
        if names:
            text, source, bucket = face_greeting.prepare(sorted(names))
            result = await voice_mcp._speak_tool(
                text, FACE_ROOM, bypass_quiet=True)
            logger.info(f"face events: spoke [{bucket}/{source}] {text!r} "
                        f"in {FACE_ROOM}: {result}")
            try:
                parsed = json.loads(result)
                if parsed.get("error"):
                    text = ""            # do not claim a greeting that failed
            except ValueError:
                text = ""
        else:
            logger.info(f"[TTS] suppressed: {', '.join(sorted(set(r[0] for r in recognized) & TTS_SUPPRESS_WHEN_ALONE))} alone — Telegram still sent")
        await self._telegram_notification(recognized, event_prefix, text)

    async def _send_photo(self, frame_path: str, caption: str, tag: str,
                          choices: list[str] | None = None) -> None:
        if not TG_TOKEN:
            return
        markup = None
        if FACE_REVIEW_ENABLED and choices:
            import secrets
            rid = secrets.token_hex(6)
            face_review.save_record(REVIEW_DIR, rid, frame_path, choices, tag)
            row = [{"text": c[:30], "callback_data": face_review.make_callback(rid, c)}
                   for c in dict.fromkeys(choices)]
            markup = {"inline_keyboard": [row[i:i + 3]
                                          for i in range(0, len(row), 3)]}
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        for attempt in range(1, TG_RETRIES + 1):
            try:
                with open(frame_path, "rb") as fh:
                    photo = fh.read()
                data = aiohttp.FormData()
                data.add_field("chat_id", TG_CHAT_ID)
                data.add_field("photo", photo,
                               filename=os.path.basename(frame_path),
                               content_type="image/jpeg")
                data.add_field("caption", caption)
                data.add_field("parse_mode", "HTML")
                if markup:
                    data.add_field("reply_markup", json.dumps(markup))
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            url, data=data,
                            timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            logger.info(f"[TG] {tag}: sent")
                            return
                        logger.warning(f"[TG] {tag}: HTTP {resp.status}: "
                                       f"{(await resp.text())[:200]}")
                        return
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"[:120]
                if attempt == TG_RETRIES:
                    logger.warning(f"[TG] {tag}: error after {attempt}: {detail}")
                    return
                await asyncio.sleep(TG_RETRY_DELAY * attempt)

    async def _telegram_notification(self, recognized: list, event_prefix: str,
                                     greeting: str) -> None:
        if not TG_TOKEN or not recognized:
            return
        time_str = event_prefix.replace("-", ":")
        for i, (name, conf, frame_path, cam_id) in enumerate(recognized):
            if i == 0:
                all_names = ", ".join(
                    f"{html.escape(r[0])} ({r[1]:.2f}) [{r[3]}]" for r in recognized)
                caption = (f"🚪 Дверь открыта: {time_str}\n"
                           f"👤 Распознано: {all_names}\n"
                           f"💬 Приветствие: {html.escape(greeting or '—')}")
            else:
                caption = (f"👤 Тоже вошёл: {html.escape(name)} ({conf:.2f}) "
                           f"[{cam_id}]")
            await self._send_photo(frame_path, caption, name,
                                   choices=[r[0] for r in recognized] + ["unknown"]
                                   if i == 0 else None)

    async def _unknown_visitor(self, day_dir: str, event_prefix: str) -> None:
        frames = event_frames(day_dir, event_prefix)
        if not frames or not TG_TOKEN:
            return
        frame = frames[0]
        cam_id = ("camera_entrance1" if "_camera_entrance1_" in
                  os.path.basename(frame) else "camera_entrance2")
        caption = (f"🚪 Дверь открыта: {event_prefix.replace('-', ':')}\n"
                   f"👤 Распознано: неизвестный посетитель "
                   f"(лицо обнаружено, но не опознано)\n"
                   f"📷 Камера: {cam_id}\n💬 Приветствие: —")
        await self._send_photo(frame, caption, "unknown-visitor",
                               choices=sorted(KNOWN_NAMES) + ["unknown"] if KNOWN_NAMES else None)

    async def _unknown_entry(self, day_dir: str, event_prefix: str) -> None:
        frame = None
        for cam_id in CAMERA_IDS:
            candidate = os.path.join(day_dir, f"{event_prefix}_{cam_id}_unknown.jpg")
            if os.path.exists(candidate):
                frame = candidate
                break
        if not frame or not TG_TOKEN:
            return
        caption = (f"🚪 Дверь открыта: {event_prefix.replace('-', ':')}\n"
                   f"👤 Распознано: неизвестно "
                   f"(лицо не обнаружено ни на одной камере)\n"
                   f"💬 Приветствие: —")
        await self._send_photo(frame, caption, "unknown-entry",
                               choices=sorted(KNOWN_NAMES) + ["unknown"] if KNOWN_NAMES else None)

    async def _save_pending(self, day_dir: str, event_prefix: str,
                            recognized: list) -> None:
        frames = event_frames(day_dir, event_prefix)
        if not frames or not recognized:
            return
        names = {r[0] for r in recognized}
        async with aiohttp.ClientSession() as session:
            for frame_path in frames[:MAX_FRAMES_PER_EVENT]:
                for face in await self._recognize_frame(
                        session, frame_path, CONTINUAL_ENROLLMENT_THRESHOLD):
                    name = face.get("name", "unknown")
                    conf = float(face.get("confidence", 0) or 0)
                    if name in names and conf >= CONTINUAL_ENROLLMENT_THRESHOLD:
                        person_dir = os.path.join(PENDING_DIR, name)
                        os.makedirs(person_dir, exist_ok=True)
                        dst = os.path.join(person_dir, os.path.basename(frame_path))
                        if not os.path.exists(dst):
                            shutil.copy2(frame_path, dst)
                            logger.info(f"[ENROLLMENT] {name}: "
                                        f"{os.path.basename(frame_path)} "
                                        f"(conf={conf:.3f}) -> pending/")
        if FACE_BEST_FRAMES_ENABLED:
            for name in names:
                hits = [(float(r[1] or 0), r[2]) for r in recognized
                        if r[0] == name and len(r) > 2]
                if hits:
                    await maybe_curate_best(name, hits)

    # --- main loop ---------------------------------------------------------
    async def run(self) -> None:
        while True:
            try:
                for day in date_dirs()[:3]:
                    day_dir = os.path.join(CAPTURES_DIR, day)
                    faces, unknown = find_new_events(day_dir, self.processed_events)
                    for prefix in sorted(faces):
                        if time.time() - self.last_trigger < COOLDOWN_SECONDS:
                            continue
                        logger.info(f"[EVENT] Processing {day}/{prefix}")
                        try:
                            recognized = await self._recognize_streaming(
                                day_dir, prefix)
                            if not recognized:
                                logger.info(f"[INFO] No known faces in {prefix}")
                                await self._unknown_visitor(day_dir, prefix)
                            else:
                                asyncio.create_task(self._save_pending(
                                    day_dir, prefix, recognized))
                        except Exception as exc:
                            logger.warning(f"[ERROR] {prefix}: "
                                           f"{type(exc).__name__}: {exc} "
                                           f"- skipping this event")
                        self.processed_events.add(event_key(day_dir, prefix))
                        self.last_trigger = time.time()
                    for prefix in sorted(unknown):
                        if time.time() - self.last_trigger < COOLDOWN_SECONDS:
                            continue
                        logger.info(f"[WAIT] New unknown entry {day}/{prefix}")
                        try:
                            await self._unknown_entry(day_dir, prefix)
                        except Exception as exc:
                            logger.warning(f"[ERROR] {prefix}: "
                                           f"{type(exc).__name__}: {exc} "
                                           f"- skipping this event")
                        self.processed_events.add(event_key(day_dir, prefix))
                        self.last_trigger = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[ERROR] Face events loop: {exc}")
            await asyncio.sleep(POLL_INTERVAL)


async def post_best_frame(frame_path: str, name: str) -> bool:
    """Send one recognised frame to identity's curator. Basic creds stay server-side."""
    ident = os.getenv("IDENTITY_URL", "http://192.168.22.102:18794").rstrip("/")
    user = os.getenv("IDENTITY_ADMIN_USER", "")
    password = os.getenv("IDENTITY_ADMIN_PASSWORD", "")
    if not user or not password:
        logger.warning("[CURATE] identity admin credentials are not configured")
        return False
    with open(frame_path, "rb") as handle:
        image = handle.read()
    form = aiohttp.FormData()
    form.add_field("file", image, filename=os.path.basename(frame_path),
                   content_type="image/jpeg")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{ident}/api/profiles/{name}/face/curate"
            async with session.post(
                    url, data=form, auth=aiohttp.BasicAuth(user, password),
                    timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"[CURATE] HTTP {response.status} for {name}")
                    return False
                body = await response.json()
                # "unchanged" is the curator's HTTP 200 answer when the gallery
                # is full and the new frame is not >=10% better. It is a handled
                # no-op, not a failure: the caller must still record the hourly
                # gap, or every arrival re-POSTs and re-runs recognition.
                return body.get("status") in ("ok", "duplicate", "unchanged")
    except Exception as exc:
        logger.warning(f"[CURATE] {type(exc).__name__}: {exc}")
        return False


async def maybe_curate_best(name: str, matches: list, *, poster=None,
                            now=None, enabled=None, gap=None) -> bool:
    """One best frame per arrival, at most once an hour per person."""
    if (FACE_BEST_FRAMES_ENABLED if enabled is None else enabled) is False:
        return False
    usable = [m for m in (matches or []) if len(m) >= 2 and float(m[0] or 0) >= 0.6]
    if not usable:
        return False
    moment = time.time() if now is None else float(now)
    window = FACE_BEST_FRAME_MIN_GAP_S if gap is None else float(gap)
    last = _LAST_CURATED.get(name, -1e18)
    if moment - last < window:
        return False
    best = max(usable, key=lambda m: float(m[0]))
    call = poster or post_best_frame
    try:
        done = await call(best[1], name)
    except Exception as exc:
        logger.warning(f"[CURATE] failed for {name}: {type(exc).__name__}: {exc}")
        return False
    if done:
        _LAST_CURATED[name] = moment
    return bool(done)



