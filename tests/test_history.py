"""Short-term dialogue ring tests (services/jev-router/history.py).

Runs offline on host python: pure stdlib, no FastAPI/aiohttp — history.py is
deliberately free of the router's network stack so this contract can be
checked without a container (field case 28.09.2026: L2 had no idea what
«выключи её» referred to because nothing kept the previous turns).
"""

import os
import sys
import time

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import history  # noqa: E402


def setup_function(_fn=None):
    """Every test starts from an empty ring (module-level state)."""
    history.reset()


def test_block_is_empty_without_turns():
    assert history.block("") == ""
    assert history.block("esp32") == ""


def test_block_lists_turns_oldest_first():
    history.push("esp32", "Включи кофеварку.", "Включила")
    history.push("esp32", "Это кофеварка.", "Да, вижу.")
    block = history.block("esp32")
    assert block.splitlines()[0].endswith("«Включила»")
    assert "«Это кофеварка.»" in block and "«Да, вижу.»" in block


def test_satellites_do_not_share_history():
    history.push("kitchen", "Выключи свет.", "Выключила")
    assert history.block("esp32") == ""
    assert history.block("kitchen") != ""


def test_push_ignores_an_empty_key():
    history.push("", "Включи кофеварку.", "Включила")
    assert history.block("") == ""


def test_ring_keeps_only_max_turns():
    for i in range(history.MAX_TURNS + 3):
        history.push("esp32", f"реплика {i}", f"ответ {i}")
    block = history.block("esp32")
    lines = block.splitlines()
    assert len(lines) == history.MAX_TURNS
    assert f"«реплика {history.MAX_TURNS + 2}»" in lines[-1]


def test_stale_turns_are_not_context():
    """An hour-old exchange must not answer «выключи её» — stale state is
    worse than no context at all."""
    history.push("esp32", "Включи кофеварку.", "Включила")
    ring = history._store["esp32"]
    ring[0] = (time.time() - history.TTL_S - 1, "Включи кофеварку.", "Включила")
    assert history.block("esp32") == ""
    assert "esp32" not in history._store  # expired ring is dropped


def test_cold_satellite_is_evicted():
    for i in range(history._MAX_STREAMS):
        history.push(f"stream{i}", "текст", "ответ")
    assert len(history._store) == history._MAX_STREAMS
    history.push("newcomer", "текст", "ответ")
    assert "stream0" not in history._store  # least recently updated goes first
    assert "newcomer" in history._store
    assert len(history._store) == history._MAX_STREAMS


def test_reset_clears_everything():
    history.push("esp32", "Включи кофеварку.", "Включила")
    history.reset()
    assert history.block("esp32") == ""
