"""The RU -> latin hint table exists twice on purpose (jev-router and
smolagents-worker are two images, each shipping its own copy), and the
copies silently drifted apart: THING resolved «робот» to domain vacuum,
but the router's _HINT_LAT had no «робот» entry, so find_entity got the RU
word as a hint, found nothing latin to match it against and answered
«Не нашла такого устройства» for a vacuum that was right there in the
registry (field case 29.09.2026, 14:17).

Both dicts are read out of the SOURCE with ast: importing the two modules
in one pytest process collides on the `config` module name (each service
has its own).
"""

import ast
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WORKER_HINTS = os.path.join(_ROOT, "services", "smolagents-worker", "ha_match.py")
_ROUTER_HINTS = os.path.join(_ROOT, "services", "jev-router", "ha_client.py")


def _literal(path: str, name: str):
    """Module-level `NAME = <literal>` from the SOURCE (any literal type)."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else list(node.targets) if isinstance(node, ast.Assign)
            else []
        )
        if any(getattr(t, "id", "") == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name!r} not found in {path}")


_dict_literal = _literal  # historical name, the tables are dicts


def test_every_worker_hint_is_known_to_the_router():
    worker = _dict_literal(_WORKER_HINTS, "HINTS")
    router = _dict_literal(_ROUTER_HINTS, "_HINT_LAT")
    missing = sorted(set(worker) - set(router))
    assert not missing, (
        "jev-router/ha_client._HINT_LAT has no expansion for "
        f"{missing} — find_entity() will return None («не нашла») for "
        "exactly the devices the resolver resolves to those stems."
    )


def test_shared_keys_resolve_to_the_same_latin_family():
    """Where both tables know the stem they must agree on the family — a
    key present on one side with an unrelated value on the other means one
    of the two services answers about the wrong entity."""
    worker = _dict_literal(_WORKER_HINTS, "HINTS")
    router = _dict_literal(_ROUTER_HINTS, "_HINT_LAT")
    shared = set(worker) & set(router)
    assert shared, "the tables do not share a single stem — sync check is broken"
    families = {
        "пылесос": {"vacuum", "roborock", "robot"},
        "робот": {"vacuum", "roborock", "robot"},
        "свет": {"light"},
        "ламп": {"light"},
        "чайник": {"kettle", "boiler"},
    }
    for stem, expected in families.items():
        if stem in worker and stem in router:
            assert set(worker[stem]) & expected, (stem, worker[stem])
            assert set(router[stem]) & expected, (stem, router[stem])


# --- The ordinal table is the same third copy of knowledge -------------------
# «в первом коридоре» has to select `corridor1_..._relay` in BOTH services:
# the router resolves it in `resolve_action` -> `find_action_targets`, the
# worker in `resolve_onoff_targets`/`match_states`. One service learning
# «второй» while the other does not would mean «первый» and «второй» behave
# differently depending on which level answered — exactly the drift this file
# exists to catch.


def test_ordinal_stems_are_identical_in_both_services():
    assert _literal(_WORKER_HINTS, "ORDINAL_STEMS") == _literal(
        _ROUTER_HINTS, "ORDINAL_STEMS"
    )


def test_ordinal_endings_are_identical_in_both_services():
    """The endings are the false-friend guard («вторник» must not be digit 2);
    they must drift together with the stems or one service starts accepting
    words the other rejects."""
    assert _literal(_WORKER_HINTS, "ORDINAL_ENDINGS") == _literal(
        _ROUTER_HINTS, "ORDINAL_ENDINGS"
    )


# --- Media services are the fourth shared table (03.10.2026) -----------------
# «поставь его на паузу» answers from the fast path (jev-router resolves
# «media__<service>» and calls the REST service directly) or from L2
# (media_control takes the service as its `action`). One level learning a
# transport verb the other does not know would mean the same command depends
# on which level answered — the same drift this file exists to catch.
_ROUTER_RESOLVER = os.path.join(_ROOT, "services", "jev-router", "resolver.py")


def test_router_media_services_are_known_to_the_worker():
    worker = _literal(_WORKER_HINTS, "MEDIA_SERVICES")
    router = _literal(_ROUTER_RESOLVER, "MEDIA_SERVICES")
    missing = sorted(set(router.values()) - set(worker))
    assert not missing, (
        f"smolagents-worker/ha_match.MEDIA_SERVICES has no {missing} — the "
        "router resolves the command and the worker would refuse the very "
        "service it produced."
    )


def test_both_levels_pick_the_same_player_for_the_same_command():
    """The choice of a target is knowledge too: «поставь ЕГО на паузу» must
    land on the same box whether L1 or L2 answered, on one live-shaped
    registry (four boxes, one of them paused, one playing, an AirPlay
    endpoint that must not count as a player)."""
    import sys

    sys.path.insert(0, os.path.join(_ROOT, "services", "smolagents-worker"))
    sys.path.insert(0, os.path.join(_ROOT, "services", "jev-router"))
    import ha_client as router  # noqa: E402
    import ha_match as worker  # noqa: E402

    states = [
        {"entity_id": "media_player.le_vlada", "state": "paused",
         "attributes": {"friendly_name": "LE-vlada"}},
        {"entity_id": "media_player.le_zal_2", "state": "playing",
         "attributes": {"friendly_name": "LE-zal"}},
        {"entity_id": "media_player.le_kitchen", "state": "idle",
         "attributes": {"friendly_name": "LE-Kitchen"}},
        {"entity_id": "media_player.tv_airplay", "state": "off",
         "attributes": {"friendly_name": "TV[AirPlay]"}},
    ]
    areas = {
        "media_player.le_vlada": "Bedroom Vlada",
        "media_player.le_zal_2": "Living Room",
        "media_player.le_kitchen": "Kitchen",
        "media_player.tv_airplay": "Living Room",
    }
    for service, name, area in [
        ("media_pause", "", ""),                    # «поставь его на паузу»
        ("media_pause", "", "гостиной"),            # a named room beats the state
        ("media_play", "", ""),
        ("volume_up", "", ""),
        ("media_next_track", "коди", "владиной комнате"),
        ("media_pause", "телевизор", "коридоре"),  # nothing there -> refuse
    ]:
        prefer = router.MEDIA_PREFER.get(service, ("playing", "buffering"))
        w = [e["entity_id"] for e in worker.resolve_media_targets(
            states, name=name, area=area, area_map=areas, service=service)]
        r = [e["entity_id"] for e in router.find_media_targets(
            states, areas, name, area, prefer)]
        assert w == r, (service, name, area, w, r)
    assert worker.MEDIA_PREFER == router.MEDIA_PREFER, (
        "the state a request is ABOUT (playing vs paused) decides which box a "
        "pronoun means — the two copies must not order it differently"
    )


def test_media_fingerprints_are_identical_in_both_services():
    """Both levels decide «did the box move?» from this tuple, and they read
    it from the same registry — a field added on one side only would make L1
    call a real volume change a failure and L2 call it a success."""
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "services", "smolagents-worker"))
    sys.path.insert(0, os.path.join(_ROOT, "services", "jev-router"))
    import ha_client as router  # noqa: E402
    import ha_match as worker  # noqa: E402

    samples = [
        {"state": "playing", "attributes": {"volume_level": 0.52,
                                             "media_title": "Jaws",
                                             "media_content_id": {"tvdb": "55730"}}},
        {"state": "idle", "attributes": {"volume_level": None,
                                          "is_volume_muted": None}},
        {"state": "paused", "attributes": {}},
    ]
    for s in samples:
        assert worker.media_fingerprint(s) == router.media_fingerprint(s)


def test_both_levels_agree_on_the_two_bedrooms():
    """This house has THREE bedrooms-as-areas: «Bedroom» (LE-spalnya) and
    «Bedroom Vlada» (le_vlada). «Сделай громче в спальне» must reach the
    first and «во владиной комнате» the second, in BOTH services.

    The failure that forced the rule: `area_matchers("Living Room")` matched
    the latin token «room» as a SUBSTRING of the stems «bedroom» and
    «bathroom», so the area became {living, bedroom, bathroom}, the exact-room
    rule preferred the phantom «Bedroom», and `media_play(area="Living Room")`
    started the show on the BEDROOM box (field check 03.10.2026 19:32).
    """
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "services", "smolagents-worker"))
    sys.path.insert(0, os.path.join(_ROOT, "services", "jev-router"))
    import ha_client as router  # noqa: E402
    import ha_match as worker  # noqa: E402

    states = [
        {"entity_id": "media_player.le_spalnya", "state": "idle",
         "attributes": {"friendly_name": "LE-spalnya"}},
        {"entity_id": "media_player.le_vlada", "state": "idle",
         "attributes": {"friendly_name": "LE-vlada"}},
        {"entity_id": "media_player.le_zal_2", "state": "idle",
         "attributes": {"friendly_name": "LE-zal"}},
    ]
    areas = {
        "media_player.le_spalnya": "Bedroom",
        "media_player.le_vlada": "Bedroom Vlada",
        "media_player.le_zal_2": "Living Room",
    }
    assert "bedroom" not in worker.area_matchers("Living Room")
    for room, want in [("спальне", ["media_player.le_spalnya"]),
                       ("Living Room", ["media_player.le_zal_2"]),
                       ("гостиной", ["media_player.le_zal_2"]),
                       ("владиной комнате", ["media_player.le_vlada"]),
                       ("Bedroom Vlada", ["media_player.le_vlada"]),
                       ("кухне", []),
                       ("коридоре", [])]:
        w = [e["entity_id"] for e in worker.resolve_media_targets(
            states, name="", area=room, area_map=areas, service="volume_up")]
        r = [e["entity_id"] for e in router.find_media_targets(
            states, areas, "", room, ("playing", "buffering", "paused"))]
        assert w == r == want, (room, w, r)


def test_both_levels_wait_the_same_time_before_calling_a_call_a_failure():
    """A volume change never reaches HA's `changed` list and lands late on an
    idle Kodi (le_spalnya 0.80 -> 0.85 arrived AFTER the 2.1 s transport window,
    field check 04.10.2026 21:08). One level waiting 2 s and the other 6.8 s is
    how the same command turns into «сделала громче» at L1 and «громкость не
    изменилась» at L2 (or the other way round) — the fingerprint is shared, so
    the schedule must be too."""
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "services", "smolagents-worker"))
    sys.path.insert(0, os.path.join(_ROOT, "services", "jev-router"))
    import ha_client as router  # noqa: E402
    import ha_match as worker  # noqa: E402

    assert worker.MEDIA_CONFIRM_DELAYS == router.MEDIA_CONFIRM_DELAYS
    assert worker.MEDIA_VOLUME == router.MEDIA_VOLUME
    assert worker.volume_unverifiable({"attributes": {"volume_level": None}})
    assert not worker.volume_unverifiable({"attributes": {"volume_level": 0.0}})
    for service in ("media_pause", "volume_up", "volume_mute"):
        assert worker.confirm_delays(service) == router.confirm_delays(service)
    # Volume gets the long window, transport the short one, in BOTH copies.
    assert sum(router.confirm_delays("volume_up")) > \
        sum(router.confirm_delays("media_pause"))
