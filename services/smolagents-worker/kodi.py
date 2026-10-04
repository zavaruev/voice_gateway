"""Kodi library access over JSON-RPC (pure helpers + the tiny HTTP client).

WHY THIS EXISTS
  «Включи следующую серию "Темного зеркала"» used to be impossible for the
  assistant: Home Assistant's MCP server exposes no media intent at all
  (see tools.MCP_TOOLS), and nothing in the stack can search a media LIBRARY.
  The model then invented a reason for the failure («не смог связаться с
  телевизором», field case 03.10.2026 18:21). The boxes themselves are quite
  capable — Kodi's JSON-RPC answers `VideoLibrary.GetTvshows` and plays an
  item with `Player.Open` — so the capability lives here and the honest
  refusal becomes «в библиотеке нет такого сериала» (verified: the household
  library has 14 Jellyfin-backed shows and no "Dark"/«Темное зеркало»).

WHY NOT THROUGH HA
  `media_player.play_media` on the Kodi integration would have to round-trip a
  Jellyfin plugin URL; `Player.Open` with the library item id is exact and
  needs no URL guessing. The RESULT is still verified through HA — the
  entity's state is what the user sees and what the honesty veto reads.

IDENTITY, NOT A ROOM TABLE
  A box is matched to its HA entity by its own zeroconf friendly name
  (`System.FriendlyName` -> "Kodi (LE-zal)" == friendly_name "LE-zal"), so
  there is no room->IP table to keep in sync and an IP change only needs the
  KODI_HOSTS seed list.

No smolagents import: tests/test_kodi.py runs this on the host python.
"""

from __future__ import annotations

import base64
import difflib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KODI_TTL = 3600.0  # host -> friendly name cache; the boxes do not move often
_KEEPALIVE = KODI_TTL * 2


def norm(s: str) -> str:
    """Comparable form of a title: lowercase, alphanumerics only.

    «Темного зеркала» -> «темногозеркала», "The Simpsons" -> «thesimpsons».
    Case, punctuation and the Latin/Cyrillic mix of a dubbed title must not
    decide whether a show is found.
    """
    return re.sub(r"[^0-9a-zа-яё]+", "", (s or "").lower().replace("ё", "е"))


def score_title(wanted: str, title: str) -> float:
    """0..100 — how well `title` answers `wanted`."""
    w, t = norm(wanted), norm(title)
    if not w or not t:
        return 0.0
    if w == t:
        return 100.0
    if t.startswith(w) or w.startswith(t):
        return 80.0 + 15.0 * min(len(w), len(t)) / max(len(w), len(t))
    if w in t or t in w:
        return 60.0 + 20.0 * min(len(w), len(t)) / max(len(w), len(t))
    ratio = difflib.SequenceMatcher(None, w, t).ratio()
    return ratio * 55.0  # fuzzy tail, below the contains tier


# Below this a "match" is noise: the household library has 14 titles, so a
# wrong pick is always available and always wrong.
MATCH_MIN = 45.0


def best_title(wanted: str, titles: list[str]) -> tuple[str | None, float]:
    """(best title, score) or (None, best score) — duplicates collapse."""
    best, score = None, 0.0
    seen = set()
    for raw in titles:
        title = (raw or "").strip()
        if not title:
            continue
        key = norm(title)
        if not key or key in seen:
            continue  # the library lists "The Simpsons" twice
        seen.add(key)
        s = score_title(wanted, title)
        if s > score:
            best, score = title, s
    return (best if score >= MATCH_MIN else None), score


def pick_episode(episodes: list[dict], want: str = "") -> dict | None:
    """The EXPLICITLY requested episode, or None when it is not in the library.

    Never substitutes a different episode: «включи 13 серию 9» on a library
    that only holds season 1 answered with S01E01 and said nothing
    (field check 03.10.2026), which is a wrong side effect wearing the
    requested episode's name. None makes the caller refuse and report.
    Use `next_episode()` for «следующая серия».
    """
    if not episodes or not (want or "").strip():
        return None
    want_n = norm(want)
    for ep in episodes:
            s, e = ep.get("season"), ep.get("episode")
            if not (isinstance(s, int) and isinstance(e, int)):
                continue
            # «S13E09», «13x9», «s13.9» and a bare «139» all reach the same
            # episode — the user does not type the canonical spelling.
            tags = {norm(f"s{s:02d}e{e:02d}"), norm(f"s{s:02d}e{e}"),
                    norm(f"s{s:02d}{e:02d}"), norm(f"s{s:02d}{e}"),
                    norm(f"s{s}e{e}"), norm(f"s{s}e{e:02d}"),
                    norm(f"s{s}x{e:02d}"), norm(f"s{s}x{e}"),
                    norm(f"{s}x{e:02d}"), norm(f"{s}x{e}"), norm(f"{s}{e}")}
            if want_n in tags:
                return ep
    # a bare episode number, spoken as a word or a digit
    for ep in episodes:
        if isinstance(ep.get("episode"), int) and want_n == norm(str(ep["episode"])):
            return ep
    return None


def next_episode(episodes: list[dict]) -> dict | None:
    """The NEXT unwatched episode — what «включи следующую серию» means.

    First with playcount 0 (Kodi's own "not watched yet"), else the LAST one
    in library order when the whole show has been watched. Falls back to the
    first entry when the library carries no playcount at all.
    """
    if not episodes:
        return None
    unwatched = [e for e in episodes if not e.get("playcount")]
    if unwatched:
        return unwatched[0]
    known = [e for e in episodes
             if isinstance(e.get("season"), int) and isinstance(e.get("episode"), int)]
    if known:
        return max(known, key=lambda e: (e["season"], e["episode"]))
    return episodes[0]


def episode_range(episodes: list[dict]) -> str:
    """«S01E01—S09E12» — what the tool says when the asked episode is absent."""
    known = [e for e in episodes
             if isinstance(e.get("season"), int) and isinstance(e.get("episode"), int)]
    if not known:
        return "нет эпизодов"
    lo = min(known, key=lambda e: (e["season"], e["episode"]))
    hi = max(known, key=lambda e: (e["season"], e["episode"]))
    return f"{episode_line(lo).split(' —')[0]}—{episode_line(hi).split(' —')[0]}"


def episode_line(ep: dict) -> str:
    """«S13E09 — The Bart Wants What It Wants» for the model to quote."""
    s, e = ep.get("season"), ep.get("episode")
    code = f"S{s:02d}E{e:02d}" if isinstance(s, int) and isinstance(e, int) else "—"
    title = ep.get("title") or ep.get("label") or ""
    return f"{code} — {title}".strip(" —")


class KodiError(Exception):
    """A box answered an error, or did not answer at all.

    `kind` separates the two failures a caller must NOT mix up:
      * "offline"  — nothing answered (timeout / refused / DNS);
      * "auth"     — HTTP 401/403: the box is UP and the credentials are
        wrong. Reported as «не отвечает» this is a wrong diagnosis, and it is
        exactly what the household produced on 03.10.2026: LE-vlada answers
        kodi/kodi while three other boxes take kodi/2441, so the shared pair
        turned a working box into a «не отвечает» report.
    """

    def __init__(self, message: str, kind: str = "offline"):
        super().__init__(message)
        self.kind = kind

    @property
    def says(self) -> str:
        """The Russian phrase a refusal should use for THIS failure."""
        return ("неверный логин или пароль" if self.kind == "auth"
                else "не отвечает")


class Kodi:
    """One box: JSON-RPC over HTTP with HTTP basic auth (stdlib only)."""

    def __init__(self, host: str, user: str, password: str, timeout: float = 6.0):
        self.host = host.strip()
        self._auth = base64.b64encode(
            f"{user}:{password}".encode()).decode()
        self.timeout = timeout

    def _url(self) -> str:
        host = self.host
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/") + "/jsonrpc"
        return f"http://{host}:8080/jsonrpc"

    def rpc(self, method: str, params: dict | None = None) -> dict:
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method,
            "params": params or {},
        }).encode()
        req = urllib.request.Request(self._url(), data=body, method="POST",
                                     headers={
                                         "Content-Type": "application/json",
                                         "Authorization": f"Basic {self._auth}",
                                     })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise KodiError(f"HTTP {e.code}",
                            "auth" if e.code in (401, 403) else "offline") from e
        except Exception as e:
            raise KodiError(str(e)[:120]) from e
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            raise KodiError(str(err.get("message", err))[:120])
        return (data or {}).get("result") or {}

    # --- identity ---------------------------------------------------------
    def friendly_name(self) -> str:
        """`System.FriendlyName` — "Kodi (LE-zal)". How we know which box."""
        labels = self.rpc("XBMC.GetInfoLabels",
                          {"labels": ["System.FriendlyName"]})
        return str((labels or {}).get("System.FriendlyName") or "")

    def short_name(self) -> str:
        """"Kodi (LE-zal)" -> "LE-zal" — matches HA's friendly_name."""
        name = self.friendly_name()
        m = re.search(r"\((.+)\)\s*$", name)
        return (m.group(1) if m else name).strip()

    def wake(self) -> bool:
        """True when the box is reachable (Kodi answers even in standby)."""
        try:
            self.rpc("JSONRPC.Ping")
            return True
        except KodiError:
            return False

    # --- library ----------------------------------------------------------
    def tvshows(self) -> list[dict]:
        res = self.rpc("VideoLibrary.GetTvshows",
                       {"properties": ["title", "year", "plot"],
                        "limits": {"start": 0, "end": 200}})
        return res.get("tvshows") or []

    def movies(self) -> list[dict]:
        try:
            res = self.rpc("VideoLibrary.GetMovies",
                           {"properties": ["title", "year", "plot"],
                            "limits": {"start": 0, "end": 200}})
        except KodiError:
            return []
        return res.get("movies") or []

    def episodes(self, tvshowid: int, limit: int = 400) -> list[dict]:
        """Episodes in library order. `playcount` is what makes «следующая
        серия» mean the next UNWATCHED one instead of the first row."""
        res = self.rpc("VideoLibrary.GetEpisodes", {
            "tvshowid": int(tvshowid),
            "properties": ["title", "season", "episode", "firstaired",
                           "playcount", "resume"],
            "limits": {"start": 0, "end": limit},
            "sort": {"orderby": "season", "sortorder": "ascending"},
        })
        return res.get("episodes") or []

    # --- playback ---------------------------------------------------------
    # Player.Open's `item` union (introspected on the boxes, Kodi 21.3 /
    # JSON-RPC 13.5): {"episodeid": N} | {"movieid": N} | {"file": path} |
    # {"playlistid": N}. The flat legacy form is the one that is accepted —
    # {"item_id": N, "video": 0} answers "Too many parameters", and
    # {"playlistid": N} alone answers "does not match any of the union types".
    # `options.resume` (introspected: bool | percentage | time) makes Kodi
    # continue from the stored resume point instead of restarting the episode —
    # which is what «включи серию» means when the user stopped halfway. For a
    # fresh episode there is no resume point and playback starts at 0.
    def play_episode(self, episode_id: int, resume: bool = True) -> dict:
        return self.rpc("Player.Open",
                        {"item": {"episodeid": int(episode_id)},
                         "options": {"resume": bool(resume)}})

    def play_movie(self, movie_id: int, resume: bool = True) -> dict:
        return self.rpc("Player.Open",
                        {"item": {"movieid": int(movie_id)},
                         "options": {"resume": bool(resume)}})


# --- the box list, probed once per TTL -------------------------------------
_BOXES: list[dict] = []
_BOXES_AT = 0.0
_BOX_LOCK = False
_KEEP: float = 0.0


def split_seed(seed: str, user: str, password: str) -> tuple[str, str, str]:
    """`"[user:pass@]host"` -> (host, user, pass), falling back to the globals.

    The credentials are per box in this house (03.10.2026): three boxes answer
    to kodi/2441 and LE-vlada to kodi/kodi. Probing the fourth with the wrong
    password made it report itself offline.
    """
    s = (seed or "").strip()
    if "@" in s:
        creds, _, host = s.rpartition("@")
        if ":" in creds:
            u, _, p = creds.partition(":")
            return host.strip(), u.strip(), p.strip()
    return s, user, password


def probe_one(seed: str, user: str, password: str, timeout: float = 6.0) -> dict | None:
    """One row of `probe_boxes`, or None for an empty seed."""
    host, u, p = split_seed(seed, user, password)
    if not host:
        return None
    box = Kodi(host, u, p, timeout=timeout)
    # ONE round trip, and every failure of it handled: an unguarded second
    # `friendly_name()` here let a half-dead box raise straight through
    # probe_boxes → boxes() → the tool, so one bad box broke the whole media
    # path instead of being reported as one bad box (caught by
    # tests/test_kodi.py::test_the_probe_does_not_spend_a_timeout_per_dead_box).
    try:
        raw = str(box.friendly_name() or "")
    except KodiError as e:
        return {"host": host, "seed": seed, "name": "", "short": "",
                "ok": False, "why": e.says, "detail": str(e)[:80]}
    except Exception as e:  # a box that answers with something unexpected
        return {"host": host, "seed": seed, "name": "", "short": "",
                "ok": False, "why": "не отвечает", "detail": str(e)[:80]}
    m = re.search(r"\((.+)\)\s*$", raw)
    short = (m.group(1) if m else raw).strip()
    # `seed` travels with the row: the credentials are per box and the caller
    # must reuse THIS box's pair, not the global default (dropping it made
    # LE-vlada answer 401 right after a successful probe).
    return {"host": host, "seed": seed, "name": raw,
            "short": short, "ok": bool(short),
            "why": "" if short else "не назвал себя", "detail": ""}


def probe_boxes(hosts: list[str], user: str, password: str,
                timeout: float = 6.0) -> list[dict]:
    """[{host, seed, name, short, ok, why, detail}] for every seed host.

    PARALLEL on purpose: the boxes are probed one at a time and a box that is
    switched off burns the FULL timeout, so one dead box silently added ~6 s to
    every media turn — on top of the three healthy ones' round trips.

    The unreachable ones are REPORTED, not dropped: «ящик во владиной комнате
    не отвечает» is a fact the user needs, and a silent omission would look
    like «нет такого сериала». `why` distinguishes an outage from wrong
    credentials, because they need different fixes.
    """
    seeds = [s for s in (h.strip() for h in hosts) if s]
    if not seeds:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(seeds))) as pool:
        rows = list(pool.map(
            lambda s: probe_one(s, user, password, timeout), seeds))
    return [r for r in rows if r is not None]


def boxes(hosts: list[str], user: str, password: str,
          timeout: float = 6.0) -> list[dict]:
    """Cached `probe_boxes` result."""
    global _BOXES, _BOXES_AT
    now = time.monotonic()
    if _BOXES and now - _BOXES_AT < KODI_TTL:
        return _BOXES
    global _BOX_LOCK, _KEEP
    if _BOX_LOCK:
        return _BOXES
    _BOX_LOCK = True
    try:
        _BOXES = probe_boxes(hosts, user, password, timeout=timeout)
        _BOXES_AT = time.monotonic()
        _KEEP = _BOXES_AT
    finally:
        _BOX_LOCK = False
    return _BOXES


def match_entity(entity_friendly: str, boxes_: list[dict]) -> dict | None:
    """The box whose own name matches the HA entity's friendly name.

    Compared normalised, so "LE-zal" matches "Kodi (LE-zal)" and
    "LE-Kitchen" matches "Kodi (LE-Kitchen)" — no room table, no IP table.
    """
    want = norm(entity_friendly)
    if not want:
        return None
    for b in boxes_:
        if b.get("ok") and norm(b.get("short", "")) == want:
            return b
    for b in boxes_:
        if b.get("ok") and want and (want in norm(b.get("short", ""))
                                     or norm(b.get("short", "")) in want):
            return b
    return None