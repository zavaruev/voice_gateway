# Kitchen scale voice commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two deterministic L1 voice commands for the ESP32 kitchen scale at 192.168.22.123 — read the weight, and tare with verification.

**Architecture:** A small pure-async MCP client module in jev-router talks to the scale's stateless `POST /mcp` endpoint (aiohttp, already a dependency). The resolver recognises the two phrasings; `_handle` intercepts them before the classifier so no LLM or easy_action fallback can reinterpret them; a dedicated `scale` execution branch speaks one honest sentence per outcome. All decision logic lives in the client module so it is tested offline without FastAPI.

**Tech Stack:** Python 3, aiohttp, FastAPI (router service), pytest, JSON-RPC 2.0 over HTTP; jev-router image built via docker socket API (no docker CLI in sandbox).

**Spec:** `docs/superpowers/specs/2026-11-22-scale-in-voice-gateway-design.md`

## Global Constraints

- Scale MCP endpoint: `POST http://192.168.22.123/mcp`, JSON-RPC 2.0, no session handshake; tools `get_weight`, `tare`, `set_unit` (only the first two used in step 1).
- `tools/call get_weight` response: `result.content[0].text` is a JSON string with `grams`, `value`, `unit`, `stable`, `zero`, `tare`, `sigma_mg`, `cal`, `tare_offset`, `uptime`.
- Default URL env: `SCALE_MCP_URL`, default `http://192.168.22.123/mcp`.
- Tare guard: refuse when `abs(grams) > 5`; after `tare`, require fresh `get_weight` with `zero == true`.
- Speech: always grams, rounded to whole grams; noun agreement грамм/грамма/граммов by last two digits.
- Router code is baked into its image: deploy = rebuild + clone-recreate through docker.sock, not `docker restart`. Gateway (`main.py`) is unchanged in step 1.
- `/route` is SSE: read every `data:` line, take the last one carrying text.
- Host shell is a sandbox with only docker.sock; use host IP 192.168.22.102 for published services.
- Code and comments in English; Russian only in user-facing strings.

## Review Focus

These are the inputs the spec implies but no single happy-path test covers; each is pinned by a test in the task named.

1. Scale unreachable / malformed RPC reply → must speak an honest refusal, never invent a weight and never escalate to L2. → Task 1 `test_unreachable_scale_speaks_refusal`.
2. `tare` issued while the platform holds an ingredient → must refuse before calling the tool, so a measured quantity is not silently zeroed. → Task 1 `test_tare_refuses_while_loaded`.
3. `tare` tool returns success but the follow-up read does not show `zero` → must NOT claim success. → Task 1 `test_tare_without_verified_zero_is_not_a_success`.
4. «обнули весы» reaching easy_action could be rewritten into a light command by the bare-verb fallback → must be intercepted before classification. → Task 2 `test_scale_phrases_are_intercepted_before_classifier`.
5. «сколько весит пылесос» / other device nouns must not be captured by the scale phrases. → Task 1 `test_scale_phrases_require_weight_words`.

---

### Task 1: Scale MCP client and speakable sentences

**Files:**
- Create: `services/jev-router/scale.py`
- Modify: `services/jev-router/config.py` (add `SCALE_MCP_URL` next to other service URLs)
- Test: `tests/test_scale.py`

**Interfaces:**
- Consumes: aiohttp (async HTTP), `config.SCALE_MCP_URL`.
- Produces: `async def get_weight(*, url=None, post=None) -> dict | None`; `async def tare(*, url=None, post=None) -> bool`; `async def tare_checked(*, url=None, post=None) -> tuple[bool, str]` (reason: `ok|unreachable|occupied|failed|unverified`); `def weight_sentence(w: dict) -> str | None`; `def sentence_for(kind: str, *, reason: str = "", weight: dict | None = None) -> str`; `async def handle(kind: str, *, url=None, post=None) -> str` where `kind` is `"weight"` or `"tare"`.
  - `post` is an injectable async callable `(url: str, payload: dict) -> dict`; default implementation performs the aiohttp POST with a short total timeout. All tests pass a stub `post`, so no device or network is needed.

- [ ] **Step 1: Write the failing tests**

`tests/test_scale.py` inserts `services/jev-router` on `sys.path` (same pattern as `tests/test_router_resolution.py`) and contains:
- `test_scale_phrases_require_weight_words`: `resolve_scale("сколько весит")` → kind `"weight"`; `resolve_scale("обнули весы")` → kind `"tare"`; `resolve_scale("сколько весит пылесос")` → None; `resolve_scale("включи свет")` → None.
- `test_weight_sentence_noun_agreement`: `weight_sentence({"grams": 0})` contains `ноль граммов`; 1 → `один грамм`; 2 → `два грамма`; 5 → `пять граммов`; 21 → `двадцать один грамм`; 22 → `двадцать два грамма`; 100 → `сто граммов`; 123 → `сто двадцать три грамма`; all prefixed `На весах`.
- `test_get_weight_parses_rpc_content`: stub `post` returns `{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\"grams\": 123.45, \"zero\": false, \"stable\": true}"}],"isError":False}}`; assert `get_weight` returns the dict with `grams == 123.45`.
- `test_unreachable_scale_speaks_refusal`: stub `post` raises `OSError`; assert `await handle("weight", post=stub)` equals the honest refusal `"Весы не отвечают."` and `await handle("tare", post=stub)` the same.
- `test_tare_refuses_while_loaded`: stub queues a loaded reading (`grams: 250, zero: false`) for the first get_weight; assert the `post` call count is 1 (no `tare` tool call) and `tare_checked` returns `(False, "occupied")`.
- `test_tare_verifies_zero_after_call`: stub responds in order: empty reading, successful tare, empty reading with `zero: true`; assert `(True, "ok")`. A fourth test with `zero: false` after tare asserts `(False, "unverified")`.

- [ ] **Step 2: Run the tests to see them fail**

Run: `pytest tests/test_scale.py -v`
Expected: FAIL — `scale` module / `resolve_scale` not defined.

- [ ] **Step 3: Implement `services/jev-router/scale.py`**

Use `num2words` (already used by `resolver._datetime_phrase`) for cardinal spelling and a local last-two-digits noun selector. Implement the JSON-RPC POST helper, the nested-text parse, a safe float conversion, and the reason-to-sentence mapping. Log every failure with `logging.getLogger("scale")`; no retries, no exception propagation to the caller. Add `SCALE_MCP_URL` to `config.py`.

- [ ] **Step 4: Add `resolve_scale` to `services/jev-router/resolver.py`**

`@dataclass class ScaleCall: kind: str` with values `"weight" | "tare"`; `def resolve_scale(text: str) -> ScaleCall | None`. Weight regex requires a weight word (`вес`, `весы`, `весит`, `весят`, `грамм`) and a question form (`сколько`, `какой`, `что показывает`); tare regex requires an imperative (`обнули`, `обнулить`, `тарируй`, `тарировать`, `сбросить ... вес`, `сделай тару`) together with `вес`/`весы`. Case-insensitive. Device-noun phrases like «сколько весит пылесос» must return None.

- [ ] **Step 5: Run the tests to see them pass**

Run: `pytest tests/test_scale.py -v`
Expected: PASS (at least 8 tests).

- [ ] **Step 6: Commit**

```bash
git add services/jev-router/scale.py services/jev-router/config.py services/jev-router/resolver.py tests/test_scale.py
git commit -m "feat(router): scale MCP client, phrases and honest verdicts"
```

### Task 2: Wire the `scale` route into `_handle`

**Files:**
- Modify: `services/jev-router/app.py` (imports, pre-classifier interception, new execution branch)
- Test: `tests/test_scale.py` (add the interception test)

**Interfaces:**
- Consumes: `resolver.resolve_scale`, `scale.handle` from Task 1.
- Produces: a new SSE route `"scale"` with `reason` `scale_weight` or `scale_tare` and exactly one `sentence` event.

- [ ] **Step 1: Write the failing test**

`test_scale_phrases_are_intercepted_before_classifier`: load `app.py` with the existing `_router_app()` helper from `tests/test_router_resolution.py`; monkeypatch `classifier.classify` to raise `AssertionError("classifier must not run")`; monkeypatch `scale.handle` to return `"На весах сто граммов."`; drive `_handle(RouteRequest(text="сколько весит", ...))` to completion and assert the first `data:` line has `route == "scale"` and a later line has the weight sentence. If `_handle` cannot be driven offline in this environment, pin it with `inspect.getsource` assertions instead (the repo already uses that pattern) and say so in the commit message.

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_scale.py::test_scale_phrases_are_intercepted_before_classifier -v`
Expected: FAIL — classifier is called / no `scale` route.

- [ ] **Step 3: Implement the interception and branch**

In `_handle`, immediately after the `normalize_stt_verbs` block and before `decision = await classifier.classify(text)`: call `resolve_scale(text)`; when it matches, set `route = "scale"`, `confidence = 1.0`, `reason = "scale_weight" | "scale_tare"`, and skip the classifier. Add `elif route == "scale":` in the execution dispatch: `sentence = await scale.handle(kind)`, append to `reply_parts`, log `speaking: %r`, and yield the `sentence` SSE event. Reuse the existing final `done` event; no new SSE types. Keep all other routes byte-identical.

- [ ] **Step 4: Run the test and the existing router suite**

Run: `pytest tests/test_scale.py -v && pytest tests/test_router_resolution.py -q`
Expected: PASS, no regressions in router resolution tests.

- [ ] **Step 5: Both-directions check**

On a throwaway copy of `app.py` with the interception call deleted, the new test must FAIL. Restore and confirm PASS. Record the result in the commit message.

- [ ] **Step 6: Commit**

```bash
git add services/jev-router/app.py tests/test_scale.py
git commit -m "feat(router): deterministic scale route before classifier"
```

### Task 3: Full suite, deploy, live verification, documentation

**Files:**
- Modify: `AGENTS.md` (append a short measured field note)
- No code files.

**Interfaces:**
- Consumes: Tasks 1–2.

- [ ] **Step 1: Run the full suite in the image**

The image bakes only a stub `/app/tests`; copy `tests/` and `services/` into a throwaway voice_gateway container via docker socket and run `python -m pytest -q`. Install `pytest pytest-asyncio httpx` together if missing (never `pytest` alone — it clobbers pytest-asyncio). Expected: all tests pass except the one documented pre-existing `test_wake_ring_buffer.py::test_ring_buffer_scores_identically` failure.

- [ ] **Step 2: Rebuild and recreate jev-router**

Use the existing `build_router.sh` / `recreate_router.sh` scripts in the opencode dir (docker socket API, no docker CLI). Keep clone env/HostConfig; health-check `http://192.168.22.102:8091/health` after start; delete the old container only after the new one answers.

- [ ] **Step 3: Live probe through /route (SSE, not JSON)**

`curl -N -s http://192.168.22.102:8091/route -H 'Content-Type: application/json' -d '{"text":"сколько весит","session_id":"probe","stream_name":"kitchen"}'` — assert a `route` event with `route=scale` and a sentence with the current weight.
Then probe `«обнули весы»` with the platform empty and confirm the spoken confirmation plus the scale's `zero: true` read-back. Do NOT tare while anything is on the platform. Record elapsed times and the exact sentences.

- [ ] **Step 4: Verify no regression in a voice turn**

Ask the kitchen camera a normal question («сколько времени») and a normal on/off command, and confirm the usual `route=easy_query` / `easy_action` events still appear. Check the router log for `route=scale` only on scale phrases.

- [ ] **Step 5: Document and commit**

Append the measured field facts (firmware v1.4.2, tool list, the two sentences, latencies, deploy method) to `AGENTS.md` as a dated entry; commit.
