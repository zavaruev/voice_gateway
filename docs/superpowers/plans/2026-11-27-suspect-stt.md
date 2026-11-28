# Suspect STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a fluent-but-wrong Whisper transcript into an answered turn with an explicit spoken "I may have misheard" cue and a correction window, instead of a 120 s silent L2 timeout.

**Architecture:** A pure resolver predicate marks utterances with no recognisable intent; the router upgrades those turns directly to Hermes with a transcript-accuracy note and appends a deterministic spoken cue; the cue's existing "повторите, пожалуйста" wording opens the camera's follow-up window so the user can say the correct version.

**Tech Stack:** Python 3.11, FastAPI/SSE (jev-router), aiohttp (CascadeBackend), pytest in-image.

**Spec:** `docs/superpowers/specs/2026-11-27-suspect-stt-design.md`

## Global Constraints

- Exact cue: `Похоже, я неверно расслышала. Если это не то — повторите, пожалуйста, правильную фразу.`
- Exact Dobbi cue: `Домовой неверно расслышал{h}. Повторите, пожалуйста, правильную фразу.` where `h` is `, сэр` / `, мэм` / `""`.
- Route reason must be exactly `stt_suspect`; the generic ack is keyed on that literal.
- No per-word repairs, no new numeric thresholds, no acoustic gates.
- Watch each test fail before implementing; verify wiring tests in both directions.

## Review Focus

- A bare verb («выключи») must keep its existing light fallback — never be marked suspect.
- A device phrase without verb («и свет») must keep the short clarification, not Hermes.
- «выключи свет» and «пауза» are actions, not suspect text.
- The cue must be the LAST sentence so the follow-up window opens after playback.
- `CAMERA_FOLLOWUP=none` remains a user opt-out; no forced window in that mode.

---

### Task 1: Pure suspect predicate

**Files:** `services/jev-router/resolver.py`, `tests/test_suspect_stt.py`

- [ ] **Step 1: Write failing tests** in `tests/test_suspect_stt.py`: `is_unrecognised_utterance("так звали Пушкина") is True`; False for `как звали Пушкина?`, `выключи свет`, `пауза`, `какой фильм мы смотрели?`, `и свет`.
- [ ] **Step 2:** Run `node /mnt/media/docker-compose/opencode/docker_run_exec.js ai-prod-voice_gateway:latest vg_pytest.sh -q tests/test_suspect_stt.py` — FAIL: function missing.
- [ ] **Step 3:** Implement `is_unrecognised_utterance(text: str) -> bool` in `resolver.py` near `is_question_shape`, using `normalize_stt_verbs`, `is_question_shape`, `knowledge_question`, `_DEVICE_NOUN` and the action-verb regexes.
- [ ] **Step 4:** Run the same command — PASS.
- [ ] **Step 5:** Commit (no git repo at root; commit in `/mnt/media/docker-compose/opencode` only if that is where version control lives — otherwise record the change in the session).

### Task 2: Router routes suspect turns to Hermes and speaks the cue

**Files:** `services/jev-router/app.py`, `services/jev-router/chat_proxy.py`, `tests/test_suspect_stt.py`

- [ ] **Step 1: Write failing tests**: router `_handle` with classifier stub returning `complex_logic/ambiguous(0.00<0.55)` and text «так звали Пушкина» yields `route=expert reason=stt_suspect`, no `_run_worker` call, `chat_proxy.stream_chat` called with `stt_suspect=True`, and the final SSE sentence is the exact cue. Guard tests: «как звали Пушкина?» and «выключи свет» do not get `stt_suspect`. `dobbi_style(cue, "сэр")` returns the exact masculine string. `stream_chat(..., stt_suspect=True)` puts the accuracy note into the system prompt.
- [ ] **Step 2:** Run the tests — FAIL on missing wiring.
- [ ] **Step 3:** Add `_STT_SUSPECT_ACK` constant, `dobbi_style` whitelist entry, suspect upgrade in `_handle`, `stt_suspect` kwarg in `chat_proxy.stream_chat` plus system note, and post-stream cue emission in the expert branch.
- [ ] **Step 4:** Run tests — PASS; then mutation-check: delete the upgrade / the cue emission in a throwaway copy and watch exactly the corresponding tests fail.
- [ ] **Step 5:** Record change.

### Task 3: Gateway suppresses the generic ack and keeps the correction window

**Files:** `backends.py`, `tests/test_suspect_stt.py`

- [ ] **Step 1:** Write failing tests: `CascadeBackend` on a route event `reason=stt_suspect` does not queue `ACK_COMPLEX`; on `reason=unresolved_question` it still does. `_is_question_reply(_STT_SUSPECT_ACK) is True`.
- [ ] **Step 2:** Run tests — FAIL.
- [ ] **Step 3:** In `backends.py`, skip the ack when `ev.get("reason") == "stt_suspect"`; other routes unchanged.
- [ ] **Step 4:** Run tests — PASS; mutation-check by restoring the ack on suspect and watching the test fail.
- [ ] **Step 5:** Record change.

### Task 4: Full verification and deployment

- [ ] **Step 1:** Full in-image suite: `node docker_run_exec.js ai-prod-voice_gateway:latest vg_pytest.sh -q tests/`; report every failure by name, including pre-existing ones.
- [ ] **Step 2:** Rebuild router image and clone-recreate the container with an image probe asserting the new symbols (`stt_suspect`, `_STT_SUSPECT_ACK`, `is_unrecognised_utterance`), health-gated delete of the old container.
- [ ] **Step 3:** Restart gateway (bind-mounted `backends.py`), check `/health` and all five camera sessions start.
- [ ] **Step 4:** Live `/route` probe with the exact field text «так звали Пушкина»: expect `expert/stt_suspect`, a Hermes answer, and the spoken cue as the last sentence. Confirm no L2 progress events and no 120 s timeout. Do not play sound in any room without an explicit instruction.
- [ ] **Step 5:** Report timings in MSK and the known residual: a real spoken correction has not been exercised through the camera.
