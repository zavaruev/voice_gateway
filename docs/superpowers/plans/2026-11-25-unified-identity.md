# Unified identity service (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One identity API and web interface that own person profiles and capture voice samples from a camera's microphone, with the two existing engines unchanged and the gateway's current paths preserved behind an empty env switch.

**Architecture:** A new small FastAPI service (`ai-prod/identity`, host port 18794) stores profile metadata in one atomic JSON file and drives the existing speaker-id (:8001) and face (:8090) APIs through their own HTTP endpoints — it never writes `.npy` directly. The gateway gains a thin identity client with a 30 s cache and a capture endpoint that taps an existing camera session's raw audio; `IDENTITY_URL=""` leaves every current call site byte-for-byte unchanged.

**Tech Stack:** Python 3.11, FastAPI, aiohttp, uvicorn, pytest / starlette TestClient; no GPU or ML dependencies in the new service.

**Spec:** `docs/superpowers/specs/2026-11-25-unified-identity-design.md` (read it first; it carries the builder-review answers).

## Global Constraints

- `IDENTITY_URL=""` ⇒ legacy behaviour exactly; no new call site is reached.
- Identity service: `network_mode: host`, port **18794** (verified free); reach it at `http://192.168.22.102:18794`.
- Two independent secrets, both fail-closed when empty: `IDENTITY_TOKEN` (Bearer, gateway→identity), `IDENTITY_ADMIN_USER`/`IDENTITY_ADMIN_PASSWORD` (Basic, UI/writes). Never reuse gateway Basic, `MCP_TOKEN` or `FACE_GREETING_TOKEN`; never commit secrets.
- Voice: `POST :8001/enroll` needs `X-API-Key` from the existing gitignored `speaker_id_service/.env` (`API_KEY`), form `user_id` + ≤50 `files`, and **replaces** the centroid — no append, no delete. Face: `POST :8090/enroll` appends (`replace=false`); the running face server reloads galleries on latest **mtime** via `reload_if_changed()`, so writes must go through its API.
- Never write a `.npy` directly. No camera state-changing action (WiFi, restart, config write, reboot, direct `/play_audio`).
- **No enrollment during this work.** `speakers/alexander.npy` must not be touched; do not enroll from `utterance_samples/*.wav` (room recordings with wake words/TV). Clean samples are recorded by the user later through the UI.
- `person` stays the engine key (`alexander`, `marina`, `vladislava`); display name is decoration only. Unknown uid ⇒ honorific `""`, never a guess; `testuser`/`testcurl` are not imported as profiles.
- Identity client cache TTL = **30 s**; profile edits must take effect without recreating the gateway.
- Sandbox has only docker.sock (no docker CLI); create/recreate containers through the socket API. BusyBox `grep` has no `\|` alternation in basic regex — use the ripgrep tool.
- Baseline gateway suite: **1002 passed, 1 skipped, 1 known fail** (`test_ring_buffer_scores_identically`). No new failures allowed.
- No GitHub push until the user confirms field behaviour.
- This workspace is **not a git repo**; every "Commit" step below is a checkpoint: confirm the listed files are saved, do not run `git` commands.

## Review Focus

- **Identity down while `IDENTITY_URL` is set** → one warning and a fallback to the current local/engine path for that turn, never silence or a wrong address. Test: Task 4, `test_identity_failure_raises_and_callers_fall_back`.
- **Voice enrollment replaces the whole centroid** → UI says so plainly; no test or verification step calls voice enroll. Test: Task 8, `test_record_endpoint_reports_replacement_and_surfaces_engine_errors`.
- **Unknown gallery user** (`testuser`) → no honorific, no profile import, no crash. Test: Task 1, `test_seed_ignores_test_profiles` and Task 4, `test_unknown_person_has_no_honorific`.
- **Direct `.npy` write** → the running face server would keep its stale mtime and ignore new samples. Test: Task 2, `test_all_writes_go_through_engine_http`.
- **Hand-built test sessions** (`CameraSession.__new__`) → a new attribute read inside the live audio path becomes `AttributeError`, which the broad `except` turns into a playback fault. Test: Task 5, `test_hand_built_sessions_carry_honorific_fields`.

---

### Task 1: Profile store and migration seed

**Files:** Create `identity/profiles.py`; Test `identity/tests/test_profiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `empty_store() -> dict`, `load_store(path: str) -> dict`, `save_store(path: str, store: dict) -> None` (atomic, temp file in the same directory + `os.replace`), `seed_store(names: dict, honorifics: dict, chat_ids: dict, voice_names: list[str], face_names: list[str]) -> dict`, `upsert_profile(store: dict, key: str, **fields) -> dict`, `profile_view(store, key, voice_names, face_names) -> dict | None`, `chat_map(store: dict) -> dict[str, str]`. Schema `{"version": 1, "profiles": {key: {display_name, honorific, telegram_chat_ids, tts_voice, notes, updated}}}`.

- [ ] **Step 1: Write the failing tests**

`test_seed_ignores_test_profiles` (three keys alexander/marina/vladislava from `honorifics`, `testuser`/`testcurl` absent from the voice gallery list), `test_atomic_save_leaves_no_partial_file` (monkeypatch `os.replace` to raise; original file unchanged), `test_chat_map_uses_confirmed_ids` (543867556→alexander, 1620243752→marina, 5136076293→vladislava), `test_unknown_profile_returns_none`, `test_corrupt_store_raises_instead_of_wiping`.

- [ ] **Step 2: Run them** — `cd identity && pytest tests/test_profiles.py -q`; expected FAIL (module missing).
- [ ] **Step 3: Implement `profiles.py`** with the signatures above; a missing file loads as `empty_store()`, a corrupt one raises.
- [ ] **Step 4: Run tests** — expected PASS.
- [ ] **Step 5: Commit** `identity/profiles.py`, `identity/tests/test_profiles.py`.

### Task 2: Engine HTTP clients

**Files:** Create `identity/engines.py`; Test `identity/tests/test_engines.py`

**Interfaces:**
- Consumes: `SPEAKER_ID_URL` (default `http://192.168.22.102:8001`), `FACE_API_URL` (default `http://192.168.22.102:8090`), `API_KEY` env.
- Produces: `EngineError(status: int, detail: str)`; `Engines(voice_url, face_url, api_key, timeout=20.0)` with `async identify(audio: bytes) -> tuple[str, float]`, `async recognize(image: bytes, threshold: float) -> list[dict]`, `async enroll_voice(key: str, files: list[tuple[str, bytes]]) -> dict`, `async enroll_face(key: str, files: list[tuple[str, bytes]]) -> dict`, `async rosters() -> tuple[list[str], list[str]]`.

- [ ] **Step 1: Write failing tests** with a stub that records `(method, url, form_fields, headers)`: `test_all_writes_go_through_engine_http` (voice `/enroll` carries `X-API-Key` + `user_id`; face `/enroll` carries `name` and no direct filesystem write), `test_identify_passthrough`, `test_recognize_passthrough`, `test_rosters_from_voice_gallery_and_face_health`, `test_non_200_raises_engine_error_with_status`.

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement `engines.py`** with aiohttp `FormData`, short total timeout, no retries, errors raised with status+body.
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit**.

### Task 3: Identity API, auth and web interface

**Files:** Create `identity/app.py`, `identity/static/index.html`, `identity/Dockerfile`, `identity/docker-compose.yml`, `identity/requirements.txt`, `identity/.env.example`, `identity/.gitignore`; Test `identity/tests/test_app.py`

**Interfaces:**
- Consumes: Task 1 store, Task 2 `Engines`.
- Produces: FastAPI `app`; `GET /health`, `GET /api/profiles`, `POST /api/profiles/{key}`, `POST /api/profiles/{key}/voice/enroll`, `POST /api/profiles/{key}/face/enroll`, `POST /api/identify` (multipart `file`), `POST /api/recognize` (multipart `file`, `threshold`), `POST /api/record` (added in Task 8), `GET /` (UI).

- [ ] **Step 1: Write failing tests** (TestClient, `Engines` stubbed): `test_empty_secrets_fail_closed` (401 with no token/password), `test_identify_joins_person_and_honorific`, `test_unknown_uid_gives_empty_honorific`, `test_enroll_error_is_returned_with_status` (no silent success), `test_ui_has_voice_enroll_and_camera_record_forms`.

- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement `app.py`**: two auth dependencies, profile endpoints, join logic, static page. The UI page lists people with voice/face status, Telegram ids and honorific; forms: create/edit profile, upload voice files (visible warning: **заменяет текущий голосовой образец**), upload photos, and `Записать с камеры` (room select + seconds, posts `/api/record`). It speaks nothing and touches no camera itself.
- [ ] **Step 4: Add packaging**: Dockerfile (`python:3.11-slim`, pip requirements, uvicorn binds `0.0.0.0:18794`), docker-compose (`network_mode: host`, **no `ports:` section**, bind mounts `data/`, galleries `:ro`, `env_file` pointing at speaker-id's `.env`), requirements (`fastapi`, `uvicorn`, `aiohttp`, `python-multipart`, `pytest`), `.env.example` with the three secret names and no values, `.gitignore` for `.env`/`data`.
- [ ] **Step 5: Run tests** — expected PASS; then build the image and check `GET /health` through docker.sock.
- [ ] **Step 6: Commit**.

### Task 4: Gateway identity client with cache and fallback

**Files:** Create `voice_gateway/identity_client.py`; Test `voice_gateway/tests/test_identity_client.py`

**Interfaces:**
- Consumes: `IDENTITY_URL`, `IDENTITY_TOKEN` env.
- Produces: `IdentityError`; `IdentityClient(url: str, token: str, ttl: float = 30.0)` with `async profiles() -> dict`, `async identify(audio: bytes) -> tuple[str, str]` (person, honorific), `async recognize(image: bytes, threshold: float) -> list[dict]`, `chat_map() -> dict[str, str]`, `honorific_for(person: str) -> str`; module-level `IDENTITY: IdentityClient | None` and `init_identity() -> None` (empty URL ⇒ `None`).

- [ ] **Step 1: Write failing tests**: `test_empty_url_keeps_identity_none`, `test_cache_refreshes_after_ttl` (fake clock), `test_identity_failure_raises_and_callers_fall_back`, `test_unknown_person_has_no_honorific`, `test_chat_map_from_cached_profiles`.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** with an asyncio lock, a single cached snapshot and one Bearer header; failure raises `IdentityError` with the status/reason.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**.

### Task 5: Wire both speaker-ID call sites and the honorific

**Files:** Modify `voice_gateway/camera_client.py` (`_fetch_speaker_id` at 6041, `_current_honorific` at 4293), `voice_gateway/main.py` (`fetch_speaker_id` at 1405, startup at ~4026); Test `voice_gateway/tests/test_two_wake_words.py` (or a new `test_identity_client.py` section)

**Interfaces:**
- Consumes: `IDENTITY` from Task 4.
- Produces: both `fetch_speaker_id` functions keep their `-> str` contract; `CameraSession._last_honorific` (new attribute, class-level default `""`).

- [ ] **Step 1: Write failing tests**: `test_camera_identity_honorific_wins`, `test_esp32_identity_honorific_wins`, `test_identity_failure_falls_back_to_direct_engine`, `test_hand_built_sessions_carry_honorific_fields` (add `_last_honorific = ""` to `_player_session()`, `_followup_session()`, `_make_session()` and any other `__new__` helper that reaches `_current_honorific`).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement**: on identity success, set `_last_honorific` / module-level last honorific and return the engine-key person; on any identity failure log one warning naming it and execute the existing direct POST unchanged. `_current_honorific()` prefers the identity value and falls back to `HONORIFIC_MAP` by uid.
- [ ] **Step 4: Run targeted tests, then the full gateway suite** — expected 1002+ passed, same single known fail.
- [ ] **Step 5: Commit**.

### Task 6: Wire face roster, recognition and Telegram chat map

**Files:** Modify `voice_gateway/face_events.py` (`_recognize_frame` at 310, roster warm), `voice_gateway/main.py` (`:4083` `_warm_face_greetings`, `:4424` Telegram `honorific_for_chat`); Test `voice_gateway/tests/test_identity_client.py`

**Interfaces:**
- Consumes: `IDENTITY.profiles()`, `IDENTITY.recognize()`.
- Produces: roster names for `face_greeting.warm`, chat→person map for Telegram.

- [ ] **Step 1: Write failing tests**: `test_greeting_roster_comes_from_identity`, `test_recognize_uses_identity_when_set`, `test_frigate_detect_stays_direct` (stub `session.post` and assert the `/detect` URL is still `FACE_API`), `test_telegram_honorific_reflects_ui_edit` (edit cached profile, call the lambda again, expect the new address).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement**: when `IDENTITY` is set, take roster and `/recognize` from identity with fallback to direct face API on failure; Telegram lambda consults `IDENTITY.chat_map()` (TTL cached) and falls back to `TG_PERSON`.
- [ ] **Step 4: Run targeted + full suite** — PASS with the known single fail.
- [ ] **Step 5: Commit**.

### Task 7: Gateway camera-audio capture endpoint

**Files:** Modify `voice_gateway/camera_client.py` (add `async record_sample(seconds: float) -> bytes`), `voice_gateway/main.py` (add `POST /api/identity/capture`); Test `voice_gateway/tests/test_identity_client.py`

**Interfaces:**
- Consumes: an existing `CameraSession`'s raw 16 kHz s16le feed (the same bytes `_feed_audio` receives), `IDENTITY_CAPTURE_TOKEN` env (Bearer, fail-closed).
- Produces: `record_sample(seconds: float) -> bytes` returning a valid mono 16 kHz WAV; gateway endpoint `POST /api/identity/capture` JSON `{room, seconds}` → `audio/wav`.

- [ ] **Step 1: Write failing tests**: `test_capture_requires_token`, `test_capture_finds_session_by_room`, `test_seconds_are_clamped_to_2_30`, `test_record_sample_writes_a_valid_wav` (inject 16 kHz frames into a fake session; read the header with `wave` and check rate/channels/duration).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement `record_sample`** as a bounded tap: set a capture buffer flag, collect only newly arriving chunks under the existing audio path (never a second ffmpeg consumer, never a lock that can stall the loop), stop on `seconds` or a hard timeout, drop the buffer on cancellation, and build the WAV with the stdlib `wave` module. Do not touch camera state.
- [ ] **Step 4: Implement the endpoint** with a session lookup by `stream_name`; 404 on unknown room, 400 on bad seconds, fail-closed 401 without the Bearer token.
- [ ] **Step 5: Run targeted + full suite** — PASS.
- [ ] **Step 6: Commit**.

### Task 8: Identity `/api/record` and the button's job

**Files:** Modify `identity/app.py` (add `POST /api/record`), `identity/static/index.html`; Test `identity/tests/test_app.py`

**Interfaces:**
- Consumes: Task 7 endpoint at `GATEWAY_URL` (default `http://192.168.22.102:18792`) with `IDENTITY_CAPTURE_TOKEN`; Task 2 `enroll_voice`.
- Produces: `POST /api/record` form/JSON `{room, profile, seconds}` → `{"status", "detail", "seconds", "room", "profile"}`.

- [ ] **Step 1: Write failing tests**: `test_record_captures_then_enrolls` (stub gateway returning WAV bytes, stub Engines recording the call), `test_gateway_error_is_surfaced_not_silenced`, `test_unknown_profile_refused_before_capture`, `test_response_names_replacement`.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement**: validate profile first, fetch WAV from the gateway with Bearer, then call `enroll_voice(profile, [("sample.wav", wav)])`; return a clear error with the failing status on either hop. The button in the UI posts here and displays the result verbatim.
- [ ] **Step 4: Run tests** — PASS.
- [ ] **Step 5: Commit**.

### Task 9: Deploy, migrate, verify read-only

**Files:** Create `opencode/create_identity.py`, `opencode/recreate_vg_identity.js` (or equivalent socket-API scripts); Modify `ai-prod/docker-compose.yml` (add `IDENTITY_URL`, `IDENTITY_TOKEN`, `IDENTITY_CAPTURE_TOKEN` to the gateway); `voice_gateway/AGENTS.md`

**Interfaces:** Consumes the built identity image and the gateway env; produces a running service and a seeded `profiles.json`.

- [ ] **Step 1: Write the migration script** that seeds from `speaker_names.json`, `person_honorifics.json`, the three confirmed chat ids and live engine rosters, **prints the diff and writes only after its own assertions pass**; `testuser`/`testcurl` excluded.
- [ ] **Step 2: Create the identity container** through docker.sock: host network, port 18794, `data/` + read-only gallery mounts, gitignored env_file for `API_KEY`. Verify `/health` 200 and the UI returns 401 without Basic.
- [ ] **Step 3: Recreate the gateway** with the three new env values (env change needs recreate, not restart). Verify `/health` and that `IDENTITY_URL=""` behaviour is still reachable by rollback.
- [ ] **Step 4: Verify read-only**: `GET /api/profiles` shows three people; `POST /api/identify` with one existing `utterance_samples/*.wav` returns `alexander` + `сэр` (no enrollment); `POST /api/identity/capture` for one room returns a WAV with valid header and non-silent level. **Do not call any enroll endpoint.**
- [ ] **Step 5: Run the full gateway suite in-image** (re-PUT `tests/` and `services/` after recreate; `pip install -q pytest pytest-asyncio httpx` together) and the identity suite; expected same single known fail.
- [ ] **Step 6: Record the measured numbers in `AGENTS.md`**; no GitHub push.
