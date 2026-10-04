"""Kodi library helpers: title matching, episode choice, box identity.

Runs offline on the host python — kodi.py is stdlib-only and imports no
smolagents, so the whole decision layer is testable without the container.
"""

import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
sys.path.insert(0, _WORKER)

import kodi  # noqa: E402

_LIBRARY = [
    "Black Mirror", "The Copenhagen Test", "The Crown", "FROM",
    "Heads and Tails", "Орел и Рошка", "Love, Death & Robots", "MobLand",
    "Proud", "Silo", "The Simpsons", "Stuart Fails to Save the Universe",
    "World Inside Out",
]


def test_norm_strips_case_punctuation_and_the_dubbed_split():
    assert kodi.norm("The Simpsons") == "thesimpsons"
    assert kodi.norm("Симпсоны") == "симпсоны"
    assert kodi.norm("Орел и Рошка") == "орелирошка"
    assert kodi.norm("Love, Death & Robots!") == "lovedeathrobots"
    assert kodi.norm("") == ""


def test_exact_and_containment_beat_fuzzy():
    assert kodi.score_title("Black Mirror", "Black Mirror") == 100.0
    assert kodi.score_title("black mirror", "Black Mirror") == 100.0
    assert kodi.score_title("Mirror", "Black Mirror") > 60
    assert kodi.score_title("Crown", "The Crown") > 60
    # A different language is NOT a match — this is why media_search returns
    # the whole catalogue instead of pretending a fuzzy hit.
    assert kodi.best_title("Симпсоны", _LIBRARY)[0] is None
    assert kodi.best_title("Темное зеркало", _LIBRARY)[0] is None


def test_best_title_dedupes_the_repeated_library_entry():
    """The household library lists "The Simpsons" TWICE (field check
    03.10.2026) — the tool must not answer with two rows for one show."""
    titles, seen = [], set()
    for t in _LIBRARY + ["The Simpsons"]:
        key = kodi.norm(t)
        if key in seen:
            continue
        seen.add(key)
        titles.append(t)
    assert titles.count("The Simpsons") == 1
    found, score = kodi.best_title("The Simpsons", titles)
    assert found == "The Simpsons" and score == 100.0


def test_best_title_ignores_empty_garbage():
    found, score = kodi.best_title("silo", ["", "   ", "Silo", None])
    assert found == "Silo" and score == 100.0


def test_pick_episode_honours_sxxexx_and_bare_numbers():
    eps = [
        {"season": 1, "episode": 1, "episodeid": 11, "title": "One"},
        {"season": 2, "episode": 3, "episodeid": 23, "title": "Three"},
        {"season": 13, "episode": 9, "episodeid": 139, "title": "Bart"},
    ]
    assert kodi.pick_episode(eps, "S13E09")["episodeid"] == 139
    assert kodi.pick_episode(eps, "13x9")["episodeid"] == 139
    assert kodi.pick_episode(eps, "s13.9")["episodeid"] == 139
    assert kodi.pick_episode(eps, "139")["episodeid"] == 139
    assert kodi.pick_episode(eps, "s2e3")["episodeid"] == 23
    assert kodi.pick_episode(eps, "3")["episodeid"] == 23      # bare episode no.
    assert kodi.pick_episode(eps, "99") is None   # absent -> refuse
    assert kodi.pick_episode(eps, "") is None     # empty want -> next_episode
    assert kodi.pick_episode([], "1") is None


def test_episode_line_is_speakable():
    assert kodi.episode_line({"season": 13, "episode": 9,
                              "title": "The Bart"}) == "S13E09 — The Bart"
    # No season/episode in the payload -> the label alone, still speakable.
    assert kodi.episode_line({"label": "raw label"}) == "raw label"


def test_a_box_is_matched_to_its_entity_by_its_own_name():
    """No room table and no IP table: identity comes from the box's zeroconf
    friendly name, so an IP change only touches KODI_HOSTS."""
    boxes_ = [
        {"host": "192.168.22.105", "short": "LE-zal", "ok": True},
        {"host": "192.168.22.118", "short": "LE-spalnya", "ok": True},
        {"host": "192.168.22.999", "short": "", "ok": False},
    ]
    assert kodi.match_entity("LE-zal", boxes_)["host"] == "192.168.22.105"
    assert kodi.match_entity("LE-Kitchen", []) is None
    assert kodi.match_entity("LE-spalnya", boxes_)["host"] == "192.168.22.118"
    # An offline box never matches, even by name: its `ok` is False.
    assert kodi.match_entity("LE-zal", [boxes_[2]]) is None


def test_short_name_strips_the_kodi_wrapper():  # noqa: D401
    """Real box answers: "Kodi (LE-zal)" -> "LE-zal", which is exactly HA's
    friendly_name for media_player.le_zal_2."""
    box = kodi.Kodi("192.168.22.105", "u", "p")
    for raw, want in (("Kodi (LE-zal)", "LE-zal"),
                      ("Kodi (LE-Kitchen)", "LE-Kitchen"),
                      ("Kodi", "Kodi")):
        box.friendly_name = lambda raw=raw: raw
        assert box.short_name() == want
    assert kodi.norm("LE-Kitchen") == kodi.norm("LE-kitchen")


def test_kodi_error_is_raised_not_swallowed():
    """A box that cannot be reached must RAISE, so the tool reports it — the
    failure mode that produced «не нашлось» instead of «ящик не отвечает»."""
    assert issubclass(kodi.KodiError, Exception)
    box = kodi.Kodi("127.0.0.1:1", "u", "p", timeout=0.3)
    try:
        box.rpc("JSONRPC.Ping")
        raise AssertionError("expected KodiError")
    except kodi.KodiError:
        pass

def test_pick_episode_never_substitutes_a_different_one():
    """The measured library entry holds season 1 only, so «13x9» does not
    exist — and the tool started S01E01 while saying nothing (field check
    03.10.2026). A named episode must resolve to itself or to nothing."""
    eps = [{"season": 1, "episode": 1, "episodeid": 245, "playcount": 1},
           {"season": 1, "episode": 2, "episodeid": 246, "playcount": 0}]
    assert kodi.pick_episode(eps, "13x9") is None
    assert kodi.pick_episode(eps, "") is None          # no want -> no pick
    assert kodi.pick_episode(eps, "S01E02")["episodeid"] == 246
    assert kodi.episode_range(eps) == "S01E01—S01E02"
    assert kodi.episode_range([]) == "нет эпизодов"


def test_next_episode_is_the_next_UNWATCHED_one():
    """«Включи следующую серию» means the next unwatched episode, not the
    first row of the library — Kodi's own playcount decides."""
    eps = [{"season": 1, "episode": 1, "episodeid": 11, "playcount": 1},
           {"season": 1, "episode": 2, "episodeid": 12, "playcount": 0},
           {"season": 2, "episode": 1, "episodeid": 21, "playcount": 0}]
    assert kodi.next_episode(eps)["episodeid"] == 12
    watched = [dict(e, playcount=1) for e in eps]
    # Everything watched -> the LAST one, not the first.
    assert kodi.next_episode(watched)["episodeid"] == 21
    # No playcount in the payload at all -> the library's own order.
    bare = [{"season": 3, "episode": 7, "episodeid": 37},
            {"season": 1, "episode": 1, "episodeid": 11}]
    assert kodi.next_episode(bare)["episodeid"] == 37
    assert kodi.next_episode([]) is None


def test_credentials_are_per_box_and_survive_the_probe():
    """The fourth box answered only to kodi/kodi while the other three take
    kodi/2441 (found 03.10.2026 by `le-vlada.local`). Two consequences, both
    measured: probing every box with the shared password reported LE-vlada
    «не отвечает», and — after the probe succeeded — the row must CARRY its
    seed, or the client falls back to the global pair and answers 401.

    `split_seed` is the single place that decides which pair a box uses.
    """
    assert kodi.split_seed("192.168.22.176", "kodi", "2441") == (
        "192.168.22.176", "kodi", "2441")
    assert kodi.split_seed("kodi:kodi@192.168.22.176", "kodi", "2441") == (
        "192.168.22.176", "kodi", "kodi")
    assert kodi.split_seed("  ", "kodi", "2441") == ("", "kodi", "2441")

    captured = []

    class _FakeBox(kodi.Kodi):
        def __init__(self, host, user, password, timeout=6.0):
            captured.append((host, user, password))

        def friendly_name(self):
            return "Kodi (LE-vlada)"

    orig = kodi.Kodi
    kodi.Kodi = _FakeBox
    try:
        rows = kodi.probe_boxes(["192.168.22.176", "kodi:kodi@192.168.22.176"],
                                "kodi", "2441")
    finally:
        kodi.Kodi = orig
    assert captured == [("192.168.22.176", "kodi", "2441"),
                        ("192.168.22.176", "kodi", "kodi")]  # order kept
    assert [r["seed"] for r in rows] == ["192.168.22.176",
                                         "kodi:kodi@192.168.22.176"]
    assert all(r["ok"] and r["short"] == "LE-vlada" for r in rows)


def test_a_wrong_password_is_not_reported_as_an_outage():
    """LE-vlada answers kodi/kodi while the others take kodi/2441; probing it
    with the shared pair answers 401 and was reported «не отвечает», sending
    the user after an outage that did not exist (03.10.2026). `KodiError.kind`
    is what keeps the two apart."""
    err = kodi.KodiError("HTTP 401", "auth")
    assert err.says == "неверный логин или пароль"
    assert kodi.KodiError("timed out").says == "не отвечает"
    # A 500 from the box is still an error, not a credential problem.
    assert kodi.KodiError("HTTP 500", "offline").says == "не отвечает"


def test_probe_row_says_why_a_box_is_unusable():
    class _Refusing(kodi.Kodi):
        def friendly_name(self):
            raise kodi.KodiError("HTTP 401", "auth")

    orig = kodi.Kodi
    kodi.Kodi = _Refusing
    try:
        rows = kodi.probe_boxes(["192.168.22.176"], "kodi", "2441")
    finally:
        kodi.Kodi = orig
    assert rows == [{"host": "192.168.22.176", "seed": "192.168.22.176",
                     "name": "", "short": "", "ok": False,
                     "why": "неверный логин или пароль", "detail": "HTTP 401"}]


def test_the_probe_does_not_spend_a_timeout_per_dead_box():
    """The boxes were probed one at a time, so one switched-off box added its
    FULL timeout to every media turn. They must be probed concurrently."""
    import time as _t

    class _Slow(kodi.Kodi):
        def friendly_name(self):
            _t.sleep(0.4)
            return "Kodi (LE-x)"

    orig = kodi.Kodi
    kodi.Kodi = _Slow
    try:
        t0 = _t.monotonic()
        rows = kodi.probe_boxes(["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
                                "kodi", "p", timeout=5.0)
        took = _t.monotonic() - t0
    finally:
        kodi.Kodi = orig
    assert len(rows) == 4 and all(r["ok"] for r in rows)
    assert took < 1.0, f"sequential probe: {took:.2f}s for 4 x 0.4s"


def test_player_open_carries_the_resume_option():
    """Kodi restarts an episode from the beginning unless it is told otherwise
    (options.resume, introspected on the boxes) — «включи серию» after stopping
    halfway must continue, not restart. The payload shape itself was the first
    thing to get wrong here: {"item_id":…, "video":0} answers "Too many
    parameters", the item must be nested under "item"."""
    captured = {}

    class _Box(kodi.Kodi):
        def rpc(self, method, params=None):
            captured[method] = params
            return {}

    box = _Box("192.168.22.105", "u", "p")
    box.play_episode(245)
    assert captured["Player.Open"] == {"item": {"episodeid": 245},
                                       "options": {"resume": True}}
    box.play_movie(12)
    assert captured["Player.Open"] == {"item": {"movieid": 12},
                                       "options": {"resume": True}}
    box.play_episode(245, resume=False)
    assert captured["Player.Open"]["options"] == {"resume": False}
