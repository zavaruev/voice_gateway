# Download confirmation at add time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speak «Скачивается <название>» when a requested torrent is found and added, instead of waiting for a completion announcement.

**Architecture:** The router parses a narrow shape of Russian download request, calls `torrent-updater`'s MCP tool `search_and_download` on `:8020/mcp` (the service that adds the torrent), and emits one spoken sentence from its success response. Unparseable requests keep the existing Hermes route. The gateway's total SSE budget rises so a slow browser call cannot be cut off.

**Tech Stack:** Python 3.12, aiohttp, FastAPI/SSE, pytest (run inside `ai-prod-voice_gateway:latest` with `/work` bound read-only).

**Spec:** `docs/superpowers/specs/2026-11-27-download-confirmation-design.md`

## Global Constraints

- No secrets in repo; `TORRENT_UPDATER_MCP_URL` is a host URL with no credentials.
- MCP is streamable HTTP: `initialize` → `notifications/initialized` →
  `tools/call`, with `Mcp-Session-Id` on the later calls and the tool body
  parsed from the SSE `data:` line's `result.content[0].text` JSON string.
- `torrent-updater` answers HTTP 200 even on failure: only the tool body's
  `success is True` counts. A failure before submission is definite (`None`);
  a timeout/unreadable answer after `tools/call` is `UNKNOWN` — never retry.
- Never add a second spoken sentence and never say «скачано».
- The silent `keepalive` events stay untouched. Hermes is used ONLY for
  requests the narrow parser cannot read; a parseable request that the tool
  refuses gets one short factual sentence instead of a tool-agent monologue.
- Router is baked into its image (rebuild + clone-recreate); gateway `backends.py` is bind-mounted (restart suffices).

## Review Focus

- «как скачать серию?» — a question must not become a download (Task 1 test).
- A device command containing no download verb stays on its old route (Task 1 test).
- Title without a season (a film) produces `kind="movie"`, `season=None` (Task 1 test).
- HTTP 200 with `success:false` must never produce a confirmation (Task 2 + Task 3 tests).
- A call lasting >200 s must not be cut by the gateway (Task 4 test pins the new 300 s default).

---

### Task 1: Parse the request and phrase the confirmation

**Files:**
- Create: `services/jev-router/download.py`
- Test: `tests/test_download_direct.py`

**Interfaces:**
- Produces: `@dataclass DownloadRequest(query: str, kind: str, season: int | None)`; `parse_request(text: str) -> DownloadRequest | None`; `confirmation_sentence(req: DownloadRequest) -> str`; `is_download_request` stays in `resolver.py` (imported, not redefined).

- [ ] **Step 1: Write failing parser tests**

```python
def test_the_field_phrase_parses():
    assert parse_request("скачай восьмой сезон Симпсонов") == DownloadRequest(
        query="Симпсонов", kind="series", season=8)
```

Also assert `parse_request("как скачать сезон?") is None`, `parse_request("выключи свет") is None`, and `parse_request("скачай фильм Интерстеллар") == DownloadRequest(query="Интерстеллар", kind="movie", season=None)`.

- [ ] **Step 2: Run them and watch the expected failure**

Run: `pytest tests/test_download_direct.py -q` (inside the image container)
Expected: FAIL — `ModuleNotFoundError: download` or missing `parse_request`.

- [ ] **Step 3: Implement the parser**

`download.py` holds an ordinal map for spoken 1–20 (whole-token match), a verb guard (reuse `resolver.is_download_request`), and a content-word guard. Strip the wake word if present, the leading verb and the `<ordinal> сезон` phrase; the remaining words are the query. `kind="series"` when a season/series/episode word is present, else `"movie"`. Return `None` when the query is empty or a question shape.

- [ ] **Step 4: Write the failing phrasing test**

```python
def test_confirmation_uses_the_requested_name():
    assert confirmation_sentence(
        DownloadRequest("Симпсонов", "series", 8)) == "Скачивается восьмой сезон Симпсонов."
```

Use the same ordinal map for the spoken season; a film returns «Скачивается <query>.».

- [ ] **Step 5: Implement `confirmation_sentence` and run the file green**

Expected: all Task 1 tests PASS.

---

### Task 2: Call the MCP tool with an injected HTTP post

**Files:**
- Modify: `services/jev-router/download.py`
- Modify: `services/jev-router/config.py` (add `TORRENT_UPDATER_MCP_URL`, `DOWNLOAD_TIMEOUT_S = 240`)
- Test: `tests/test_download_direct.py`

**Interfaces:**
- Produces: `async def search_and_download(req: DownloadRequest, *, url: str | None = None, post=None) -> dict | None`; `post(url, rpc_payload, headers) -> (status, text, headers)` is injected by tests, production uses aiohttp with `ClientTimeout(total=DOWNLOAD_TIMEOUT_S)`.

- [ ] **Step 1: Write failing tests with a recording stub**

The stub must mirror the real server: an SSE `data:` line for `initialize` carrying `Mcp-Session-Id`, an empty 202 for `notifications/initialized`, and an SSE `tools/call` result whose `content[0].text` is a JSON string. Assert the call sequence, the session header on later calls, and `arguments == {"query": "Симпсонов", "media_type": "series", "season": 8, "verify": False}`; assert `None` for `success:false`, a missing success field, `isError:true` and a JSON-RPC error; assert `UNKNOWN` for a timeout or malformed body after `tools/call`, and `None` for a failure before submission.

- [ ] **Step 2: Run and watch them fail** — the old client posts a raw REST payload with two arguments.

- [ ] **Step 3: Implement the client**

Use `config.TORRENT_UPDATER_MCP_URL`; parse SSE/JSON results by id; check `isError` and `success`; log the body's `torrent.title` at INFO when successful. No retry (a browser search repeated costs minutes).

- [ ] **Step 4: Run Task 1 + Task 2 tests green.**

---

### Task 3: Wire the router branch

**Files:**
- Modify: `services/jev-router/app.py` (import `download`, branch after the `download_request` routing block)
- Test: `tests/test_download_direct.py`

**Interfaces:**
- Consumes: `download.parse_request`, `download.search_and_download`, `download.confirmation_sentence`.
- Produces: the `download_request` route emits exactly one `sentence` event then the normal `done` — success, definite refusal and `UNKNOWN` each have one factual sentence. Only an unparseable request falls through to `chat_proxy.stream_chat`.

- [ ] **Step 1: Write failing router tests**

Monkeypatch `classifier.classify` to `complex_logic`, stub `download.search_and_download` to a success dict, drive `_handle` with the field text, and assert: `route.reason == "download_request"`, `sentences == ["Скачивается восьмой сезон Симпсонов."]`, `chat` not called, `worker` not called. Add the mirror test with `search_and_download` returning `None`: the chat proxy IS called and no confirmation sentence appears.

- [ ] **Step 2: Run and watch them fail** — chat is called on success today.

- [ ] **Step 3: Implement the branch**

After setting `route, reason = "expert", "download_request"`, set a local `direct_download = download.parse_request(text)`. In the `general_qa / expert` execution branch, when `direct_download` is set, call `download.search_and_download` and yield one `sentence` on success; otherwise keep the existing keepalive-wrapped chat loop. Do not emit `_STT_SUSPECT_ACK` on this path.

- [ ] **Step 4: Run the route tests and the adjacent `test_suspect_stt.py` / `test_download_keepalive.py` green.**

---

### Task 4: Give the gateway enough total budget

**Files:**
- Modify: `backends.py` (`CascadeBackend.total_timeout` default 200 → 300, comment naming the 27.11 browser call)
- Test: `tests/test_download_direct.py`

- [ ] **Step 1: Write a failing assertion on the constructor default**

`assert inspect.signature(CascadeBackend.__init__).parameters["total_timeout"].default == 300`.

- [ ] **Step 2: Run it red.**

- [ ] **Step 3: Change the default and colour comment; run the whole suite.**

Expected: the new file and all existing tests pass except the documented pre-existing `test_wake_ring_buffer.py::test_ring_buffer_scores_identically`.

- [ ] **Step 4: Commit the change set** (no git CLI on the host; record the file list in the final report instead).

---

### Task 5: Deploy and verify live

**Files:**
- Reuse `docs/superpowers/plans/...` neighbours' pattern: a throwaway container with the docker socket and `/repo:ro`, rebuild `ai-prod-jev-router:latest`, probe the new symbols in the image, stop/rename/clone-create/start, health-gate the delete of the old container. Gateway needs only a restart (`backends.py` is bind-mounted).

- [ ] **Step 1: Rebuild + clone-recreate the router; health must answer before the old container is deleted.**
- [ ] **Step 2: Restart the gateway and check `/health` and all five camera sessions.**
- [ ] **Step 3: Direct `/route` probe with the exact field text; read the SSE events and stop after `route` (do not trigger a real download).**
- [ ] **Step 4: Check Transmission for duplicates and report the state of the existing `S08` torrent.**
