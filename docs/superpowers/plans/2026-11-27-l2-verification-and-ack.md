# L2 verification and spoken ack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** L2 verifies an on/off intent with the same physics instrument as L1 and speaks a short «Включаю.»/«Выключаю.» only after the action is confirmed.

**Architecture:** A small pure module is duplicated into both service images and held identical by a sync test. Router aliases its existing private helper names to it. Worker's on/off read-back bypasses the 10 s state cache, returns `reached`/`unproven` separately, and writes a per-request ack through a `ContextVar` that the SSE poll loop emits before the model's final answer.

**Tech Stack:** Python 3.12, aiohttp/FastAPI SSE, pytest, two separate Docker images (`jev-router`, `smolagents-worker`) built via docker.sock.

**Spec:** `docs/superpowers/specs/2026-11-27-l2-verification-and-ack-design.md`

## Global Constraints

- `delays = (1.0, 1.5, 2.5)` stays literally in `confirm_on_off_reached` and `confirm_on_off_one_read`; a test compares the two schedules.
- Power sensor pairing: `switch.<base>` → `sensor.<base>_power` with `device_class == "power"`; `POWER_ON_W = 5.0`.
- The measured coffee-maker lag is +2.93 s; one extra read at +1.0 s after the three base reads is allowed, no other timing changes.
- «Включаю.» is emitted ONLY on `reached`. Never on `route`, on a progress heartbeat, or on `unproven`.
- Do not touch STT, wake, arbiter, audio, camera settings, or the L1 two-part answer shape.
- Both service Dockerfiles already use `COPY *.py ./`; no Dockerfile edit is needed.
- In-image test runs need `tests/` + `services/` copies and `pytest pytest-asyncio httpx` installed together after any container recreate.

## Review Focus

- A device that moved but whose power sensor has not moved yet → must be `unproven`, never a refusal (the 27.11 coffee maker).
- HA optimistic state `on` with power 0 W → must NOT count as `reached` (the 10.10 lie).
- A cached pre-call snapshot → must never be used as post-call evidence.
- A failed or rate-limited model after a verified tool success → ack already spoken, no apology should overwrite it.
- Parallel worker turns → ack from one turn must not leak into another.

---

### Task 1: Shared on/off verification module

**Files:**
- Create: `services/jev-router/onoff_verify.py`
- Create: `services/smolagents-worker/onoff_verify.py`
- Test: `tests/test_hint_sync.py`

**Interfaces:**
- Produces: `POWER_SENSOR_SUFFIX`, `POWER_ON_W`, `power_sensor_for(entity_id: str, states: list[dict]) -> str | None`, `physics_agrees(ent: dict | None, target: str) -> bool | None`, `ack_for_onoff(target: str) -> str`.

- [ ] **Step 1: Write the failing sync test**

```python
def test_onoff_verify_module_is_identical_in_both_services():
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    a = os.path.join(root, "services", "jev-router", "onoff_verify.py")
    b = os.path.join(root, "services", "smolagents-worker", "onoff_verify.py")
    assert open(a, encoding="utf-8").read() == open(b, encoding="utf-8").read()
```

- [ ] **Step 2: Run it and see it fail** — `pytest tests/test_hint_sync.py::test_onoff_verify_module_is_identical_in_both_services -v`; expected FAIL (file not found).

- [ ] **Step 3: Implement both files identically** — pure stdlib; port the body of router's current `_power_sensor_for`/`_physics_agrees` verbatim; `ack_for_onoff("on") == "Включаю."`, `("off") == "Выключаю."`. Add `test_power_sensor_needs_device_class` (an impostor `sensor.coffemaker_power` without `device_class: power` returns None) and `test_ack_phrases` in the same new test file.

- [ ] **Step 4: Run the three tests and confirm PASS.**

- [ ] **Step 5: Commit** — `git add ... && git commit -m "feat: shared on/off verification module"`.

### Task 2: Router uses the shared helpers without changing its public names

**Files:**
- Modify: `services/jev-router/app.py` (the local `_POWER_SENSOR_SUFFIX`/`_POWER_ON_W`/`_power_sensor_for`/`_physics_agrees` block and its imports).

**Interfaces:**
- Consumes: Task 1's `power_sensor_for`, `physics_agrees`.
- Produces: `jev_router_app._power_sensor_for` / `._physics_agrees` remain callable, and `_power_sensor_for` still appears in `confirm_on_off_one_read`'s source.

- [ ] **Step 1: Run the existing guards first** — `pytest tests/test_router_resolution.py -k 'power_sensor or physics_agrees or first_read_is_not_documented' -v`; expected PASS on current code (these are regression locks, not new tests).

- [ ] **Step 2: Replace the four local definitions with aliases** to the shared module, keeping the names `_power_sensor_for` and `_physics_agrees` and keeping the call inside `confirm_on_off_one_read` spelled exactly that way.

- [ ] **Step 3: Re-run the same tests** — expected PASS; if the source-inspection test fails, restore a local wrapper with the original name rather than weakening the test.

- [ ] **Step 4: Commit** — `git commit -m "refactor: router on/off helpers use the shared module"`.

### Task 3: Worker read-back: fresh state + power sensor + unproven

**Files:**
- Modify: `services/smolagents-worker/ha_match.py`, `services/smolagents-worker/tools.py`
- Test: `tests/test_l2_onoff_ack.py` (new)

**Interfaces:**
- Consumes: Task 1's `power_sensor_for`, `physics_agrees`.
- Produces: `confirm_on_off_reached(ids, target, fetch_states) -> list[str]` (unchanged signature), `confirm_on_off_verdict(ids, target, fetch_states) -> tuple[list[str], list[str]]`, and `tools._fresh_states() -> list | None`.

- [ ] **Step 1: Write failing tests**

```python
def test_fresh_read_sees_the_change_the_cache_hid():
    # Cached snapshot says off; the fresh fetcher says on from the first read.
    stale = [{"entity_id": "switch.coffemaker", "state": "off"}]
    fresh = [{"entity_id": "switch.coffemaker", "state": "on"}]
    assert confirm_on_off_verdict(["switch.coffemaker"], "on", lambda: fresh) == (
        ["switch.coffemaker"], []
    )

def test_optimistic_state_without_power_is_not_reached():
    rows = [{"entity_id": "switch.coffemaker", "state": "on"},
            {"entity_id": "sensor.coffemaker_power", "state": "0",
             "attributes": {"device_class": "power"}}]
    assert confirm_on_off_verdict(["switch.coffemaker"], "on", lambda: rows) == (
        [], ["switch.coffemaker"]
    )

def test_power_sensor_arriving_after_the_third_read_is_proved():
    # Measured 27.11: power moved at +2.93 s, after all three base reads.
    calls = {"n": 0}
    def _feed():
        calls["n"] += 1
        watts = "0" if calls["n"] <= 4 else "1031"
        return [{"entity_id": "switch.coffemaker", "state": "on"},
                {"entity_id": "sensor.coffemaker_power", "state": watts,
                 "attributes": {"device_class": "power"}}]
    assert confirm_on_off_verdict(["switch.coffemaker"], "on", _feed)[0] == [
        "switch.coffemaker"
    ]
```

- [ ] **Step 2: Run them and confirm they fail** on the current `ha_match` (no `confirm_on_off_verdict`).

- [ ] **Step 3: Implement** `confirm_on_off_verdict` in `ha_match.py`, keeping `confirm_on_off_reached` as a thin `-> verdict[0]` wrapper that still contains the literal `delays = (1.0, 1.5, 2.5)`. Base loop uses `fetch_states()` each pass; if a sensor exists and `physics_agrees is True` → reached; if a sensor exists and is not yet agreeing → unproven; otherwise `state == target` → reached. After the base loop, if anything is still unproven AND has a sensor, sleep 1.0 s and re-read once.

- [ ] **Step 4: Add `tools._fresh_states()`** — direct `GET {HA_URL}/api/states` with the HA token, no `_RAW_*` cache; pass it to `confirm_on_off_verdict` in `_verify_on_off_result`. Success → keep the original result; unproven → «Пока не подтвердилось…» without claiming failure; neither → the existing refusal text.

- [ ] **Step 5: Run the new tests plus `pytest tests/test_honesty.py -k 'read_back or moved or two_paths' -v`** — all PASS; the old three-read schedule test must still pass.

- [ ] **Step 6: Commit** — `git commit -m "fix: L2 on/off read-back uses fresh state and power sensor"`.

### Task 4: Worker speaks the ack before the model's final answer

**Files:**
- Modify: `services/smolagents-worker/app.py`, `services/smolagents-worker/tools.py`
- Test: in-container probe (worker image has `smolagents`, the gateway test image does not).

**Interfaces:**
- Consumes: Task 1's `ack_for_onoff`, Task 3's verdict.
- Produces: `ack_box: list[str]` per `/invoke` request; SSE `sentence` event with the ack emitted before `final_answer` sentences.

- [ ] **Step 1: Add a pure unit test for `ack_for_onoff`** (Task 1 file) — already covered; here add `test_no_ack_when_unproven` at the `tools.py` boundary using a stub verdict, asserting no ack is written.

- [ ] **Step 2: Implement the per-request channel** — a module-level `ContextVar` holding a list; `_handle` sets a fresh list and `asyncio.to_thread` inherits the context; `ha_action` appends `ack_for_onoff(target)` only on `reached`; `_handle`'s poll loop yields the pending ack as `{"type":"sentence","text":...}` once, before `box["answer"]` sentences.

- [ ] **Step 3: In-container probe** — `POST /invoke` with a stubbed HA that reports power 1031 W; assert the first SSE sentence is «Включаю.» and that a failure stub produces no ack. Write the script to a bind-mounted file and assert on its exit code (do not parse the docker-logs multiplexer).

- [ ] **Step 4: Commit** — `git commit -m "feat: L2 speaks a short ack after a verified action"`.

### Task 5: Full suite, deploy and live verification

**Files:** none (verification and release).

- [ ] **Step 1: Full in-image suite** — `pytest tests/ -q` in a fresh container; expected ≥ current baseline (1113 passed / 1 skipped at 27.11) with the known `test_ring_buffer_scores_identically` as the only accepted failure. Write output to a bind-mounted file and read it host-side; do not tally through docker logs.

- [ ] **Step 2: Rebuild both images and clone-recreate both containers** through docker.sock, health-gated; preserve env; delete old containers only after a fresh 200 and a fresh startup line.

- [ ] **Step 3: Live `/route` on the exact field text** — `включи к феварку` with `stream=kitchen`; confirm the SSE queue order: ack «Включаю.» before any model answer, no «не получилось» while power is drawn. Do not toggle a device without fresh explicit instruction; use the recorded phrase against a stub if the user has not authorized a real switch.

- [ ] **Step 4: Report MSK times** for every event quoted from the logs.
