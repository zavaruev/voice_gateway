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


def _dict_literal(path: str, name: str) -> dict:
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
