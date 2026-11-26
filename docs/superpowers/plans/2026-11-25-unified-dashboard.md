# Unified Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one web entrance that ports the gateway and identity UIs into the identity design language and adds transparent camera configuration with a safe staged diff + Apply orchestration, without touching the live gateway until verified.

**Architecture:** New small FastAPI service `ai-prod/dashboard/` on host port 18795, static single-file HTML in the identity token set, server-side proxy to gateway (:18792) and identity (:18794) using credentials from gitignored env. It holds only docker.sock + compose/`.env` access; gateway/identity APIs and their old UIs remain untouched and alive as fallback. Camera Apply is a journaled state machine with a shared lock, anchored config edits, clone-recreate via socket API, go2rtc re-registration and structured rollback.

**Tech Stack:** Python 3.11, FastAPI/uvicorn, aiohttp, httpx, pytest; plain HTML/CSS/JS (no bundler); docker socket API through `aiohttp.UnixConnector` / `httpx` with Unix socket; no new GPU dependencies.

**Spec:** `docs/superpowers/specs/2026-11-25-unified-dashboard-design.md`

## Global Constraints

- Sandbox has **no `python3`, no `git` CLI**; all tests run via `docker exec` in an image that has pytest (the established pattern in AGENTS.md). Commit steps are replaced by a progress ledger entry.
- Bind to host IP `192.168.22.102`, port 18795; never `0.0.0.0`. Basic auth fail-closed with `WWW-Authenticate: Basic realm="Dashboard"`; no token/secret ever in a response, log, diff or audit entry.
- Identity design tokens verbatim: `--blue:#0066cc`; black 44 px nav; 52 px frosted sub-nav `blur(20px) saturate(180%)`; 1200 px `minmax(320px,1fr)` grid; 18 px cards; exactly ONE box-shadow (avatars); light/dark `localStorage`; `Cache-Control: no-store`.
- Compose is authoritative for cameras; `.env` keys are merged into the ORIGINAL file on disk, never round-tripped from displayed redacted state.
- Anchored replacements must occur **exactly once**; mismatch ⇒ write nothing, report.
- No camera-side state change (no ifconfig / majestic restart / reboot / config write / direct play_audio). `config_drift` is read-only.
- Whole-gateway restart is disclosed in the typed confirmation: all rooms go briefly deaf.
- Logs: UTC/MSK labelled; health gated by container start epoch, never stale lines; docker log frames demuxed (8-byte header).
- No `import main` in any new module; P6 (separate plan) keeps top-level modules and re-exports.
- Old UIs stay alive until the user explicitly approves deletion on the agent's recommendation. No GitHub push.

## Review Focus

- **Redacted `.env` written back verbatim** would destroy credentials ⇒ `test_env_merge_never_writes_redaction` in Task 7.
- **Verbatim RTSP URL with `root:…` leaking** into provenance, diff, errors or live logs ⇒ `test_no_secret_in_provenance_or_logs` in Task 6 and Task 8.
- **Anchor matching zero or two times** silently editing the wrong block ⇒ `test_anchor_must_match_exactly_once` in Task 7.
- **Stale startup line from a previous container** passing the health gate ⇒ `test_health_uses_start_epoch` in Task 8.
- **go2rtc self-referential stream / DELETE-PUT gap** leaving a DESCRIBE-404 room ⇒ `test_stream_verification_detects_self_reference` in Task 8.

---

### Task 1: Dashboard service skeleton, auth, design shell

**Files:** Create `ai-prod/dashboard/{app.py,auth.py,config.py,static/index.html,Dockerfile,docker-compose.yml,requirements.txt,.env.example}`; create `ai-prod/dashboard/tests/test_shell.py`.

**Interfaces:**
- Produces: `create_app() -> FastAPI`; `require_admin(request) -> None` (raises 401 with challenge, 503 when credentials unset); `SETTINGS` with `gateway_url`, `identity_url`, `go2rtc_url`, `host_ip`, `port=18795`; `GET /health` (unauthenticated), `GET /` (Basic, `no-store`, 200 with nav tokens).

- [ ] **Step 1: Write failing tests** for: `GET /health` 200 without auth; `GET /` 503 when `DASH_ADMIN_USER`/`DASH_ADMIN_PASSWORD` empty; 401 with `WWW-Authenticate` on wrong Basic; 200 with correct Basic; `Cache-Control: no-store`; HTML contains the identity tokens (44 px nav, 52 px frosted sub-nav, `minmax(320px,1fr)`, `--blue:#0066cc`, dark-theme block, theme toggle) and all six nav labels (Status, Cameras, Devices, People, Settings, Services).

- [ ] **Step 2: Run them in-container** (copy `dashboard/` into an image with pytest, run `pytest tests/test_shell.py -v`). Expected: failures on missing module/routes.

- [ ] **Step 3: Implement** `config.py` (env, no secrets in repr), `auth.py` (constant-time compare, fail-closed), `app.py` (`create_app`, static mount, lifespan placeholder), `static/index.html` (single-file shell, same token block as `identity/static/index.html`, six empty page containers). `Dockerfile`: `python:3.11-slim`, `COPY app.py auth.py config.py ./`, `COPY static/ ./static/`, `pip install -r requirements.txt`, `EXPOSE 18795`. `docker-compose.yml`: `network_mode: host`, `restart: unless-stopped`, `env_file: .env.voice_gateway` for service credentials, own `ports` only if not host-network (host network ⇒ no port mapping), volumes: docker.sock, compose file (rw), `.env.voice_gateway` (ro), `voice_gateway/config` (ro), `voice_gateway/logs` (ro).

- [ ] **Step 4: Run tests** — expected all pass; record in `.superpowers/dd/2026-11-25-unified-dashboard/progress.md`.

- [ ] **Step 5: Deploy P0** — build via docker socket API (no CLI), create container, health-gate `/health`, verify `GET /` challenge. Old UIs untouched.

### Task 2: Status and Services pages (read-only)

**Files:** Modify `app.py`; create `discovery.py`, `tests/test_discovery.py`, `tests/test_status.py`.

**Interfaces:**
- Consumes: `SETTINGS`, `require_admin`.
- Produces: `async def collect_status() -> dict` with keys `gateway`, `identity`, `rooms`, `containers`, `alerts`; `GET /api/status` (Basic); `GET /api/links` (Basic) returning the service link registry (gateway, identity, router 8091, Frigate, go2rtc, face 8090, speaker-id 8001, TTS 5051/5053).

- [ ] **Step 1: Tests** with stubbed HTTP/socket clients: gateway `/health` + `/api/devices` + `/api/camera/tts` absent; identity `/health` + `/api/profiles`; go2rtc `/api/streams` producer/consumer shape; docker `/containers/json`. Assert statuses, `alerts` for a room with 0 consumers or `recv=None`, and that no credential appears in the JSON.

- [ ] **Step 2: Run** — expected failures.

- [ ] **Step 3: Implement** `discovery.py` with timeouts ≤2 s per service and honest `unknown` on failure (never "healthy" on a failed read); `alerts` names the room and the measured number; links injected server-side into the HTML (no client-side secret).

- [ ] **Step 4: Run tests + live smoke** — `/api/status` returns all five rooms and all three profiles on the real LAN.

### Task 3: Feature-parity matrix and ported People page (read)

**Files:** Create `docs/feature-parity.md`, `proxy.py`, `people.py`, `tests/test_proxy.py`, `tests/test_people.py`.

**Interfaces:**
- Consumes: `SETTINGS.gateway_url/identity_url`; service credentials loaded from env at config time.
- Produces: `async def proxy(method: str, path: str, *, body: bytes|None = None, params: dict|None = None, service: str) -> tuple[int, bytes, dict]` — server-side, never forwards browser credentials; uses gateway Basic or identity Basic+Bearer as appropriate; strips `Set-Cookie` and hop-by-hop headers. `GET /api/people` proxied from identity `/api/profiles`.

- [ ] **Step 1: Build the matrix** from real sources: identity `app.py` routes `:199-424`, identity `static/index.html` delegated events and theme toggle; gateway `main.py` routes in the REST API table, `templates/index.html` tabs, API calls at `:489-496`. Record every endpoint, auth and UI behaviour in `docs/feature-parity.md`.

- [ ] **Step 2: Tests** — proxy sends server-side creds, rejects an unknown `service`, never echoes request credentials; `/api/people` returns identity profiles with `voice_enrolled`/`face_enrolled` and **no** API key.

- [ ] **Step 3: Run** — expected failures; implement `proxy.py`/`people.py`; port identity's card grid, theme toggle, `no-store`, photo thumbnail strip into the dashboard shell.

- [ ] **Step 4: Verify** — parity rows for identity all tick; live `GET /api/people` returns three real profiles.

### Task 4: People page writes (profile/voice/face/photo/camera record)

**Files:** Modify `proxy.py`, `people.py`, `static/index.html`; `tests/test_people.py`.

**Interfaces:**
- Produces: POST/DELETE proxy methods for identity routes (`/api/profiles/{key}`, voice/face enroll, photo delete, `/api/record`), with `allowed_rooms()` seeded from the same `IDENTITY_ROOMS` list and quality verdicts passed through unchanged.

- [ ] **Step 1: Tests** — a POST creates a profile through the real endpoint shape; an invalid room is refused before any capture; photo delete sends the original-frame rebuild path; 502 from identity is surfaced with real detail, never "success"; every response redacts credentials.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: Live smoke with a THROWAWAY profile only** — create then delete `zero_test`, leaving the three real profiles intact (the established rule: do not delete real biometrics).

### Task 5: Devices / ESP page (read then write)

**Files:** Create `devices.py`; modify `static/index.html`; `tests/test_devices.py`.

**Interfaces:**
- Produces: proxies for `GET /api/devices`, `GET/POST /api/devices/config`, `PUT/DELETE /api/devices/config/{mac}`, `POST /mcp/{session_id}`, `POST /api/tts`, `GET/POST /api/firmware[/upload]`, `GET/POST /ota`, all with gateway admin Basic server-side; page ports Devices/Sessions/Control tabs (tables, MCP console, firmware upload) into the dashboard design.

- [ ] **Step 1: Tests** — device CRUD round-trip on stubs; firmware size cap forwarded; OTA handshake fields unchanged; an unknown session is refused; no gateway credential in any response; UI contains the MCP console and firmware upload controls.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: Live read-only check** — list real devices/sessions/firmware metadata; do NOT flash firmware or send to a device in this task.

### Task 6: Camera provenance and redaction (read-only)

**Files:** Create `provenance.py`, `secrets.py`; `tests/test_provenance.py`; modify `static/index.html`.

**Interfaces:**
- Consumes: compose text and `.env` text read from disk.
- Produces: `def camera_provenance(compose_text: str, env_text: str) -> dict[str, dict]` — per room: `stream`, `source_url` (REDACTED), `play_audio_url` (REDACTED), `wake_model`, `webrtc`, `tuning` (read-only), plus `source_lines` with file/line numbers and `effective_since` from container start; `def redact(text: str) -> str` covering `root:PW@`, `api_key=`, `token=`, Bearer/Basic values and RTSP userinfo.

- [ ] **Step 1: Tests** — parse the real compose fixture with five rooms; assert all five present; assert the verbatim source URL (with credentials) NEVER appears in the output or in a forced error string; assert redaction of a raw ffmpeg line containing the URL; assert a room present in `CAMERA_STREAMS` but missing its `*_<NAME>` var is reported as a gap, not guessed.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: Page** — Cameras tab shows provenance cards plus live numbers from Task 2 (chunks/s, producer B/s, play_audio ratio, rms/floor), each labelled with UTC/MSK and "effective since".

### Task 7: Staged diff engine (no Apply yet)

**Files:** Create `compose_edit.py`; `tests/test_compose_edit.py`.

**Interfaces:**
- Consumes: current compose/`.env` file paths.
- Produces: `def stage_edit(text: str, anchor: str, replacement: str) -> StagedEdit` with `.changed_text`, `.preview` (redacted unified diff), `.reject_reason`; `def env_merge(original_text: str, changes: dict[str,str]) -> str`; `def backup(path: str, now: float) -> str` (0600, non-UI dir).

- [ ] **Step 1: Tests** — anchor exactly once ⇒ staged; zero or two ⇒ reject with reason and NO changed text; redacted preview contains no secret; `env_merge` on a file containing `TOKEN=real` with a redacted UI value leaves `TOKEN=real` untouched; backup mode 0600 and name includes a UTC timestamp.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: Re-read the edited functions after implementation** (the standing rule: an assertion that an edit landed is not proof; re-read the file).

### Task 8: Apply state machine — journal, lock, docker, go2rtc, rollback

**Files:** Create `apply.py`, `docker_ctl.py`, `go2rtc.py`, `journal.py`, `audit.py`; modify `camera_watchdog.sh` (additive shared lock); `tests/test_apply.py`, `tests/test_docker_ctl.py`, `tests/test_go2rtc.py`, `tests/test_journal.py`.

**Interfaces:**
- Produces: `async def apply_staged(staged: StagedEdit, *, confirm_text: str, now: float) -> ApplyResult`; `ApplyResult` with `phase`, `changed`, `not_rolled_back`, `measurements`, `ok`. `docker_ctl`: `stop`, `rename`, `clone_create`, `start`, `health(epoch)`, `logs_since(epoch)` (8-byte demux); `go2rtc`: `snapshot(name)`, `delete(name)`, `put(name, url)`, `verify(name, expected)`; `journal`: `write_phase`, `resume_or_abort`; one global lock file shared with `camera_watchdog.sh`.

- [ ] **Step 1: Tests (on stubs, never real containers)** — typed confirmation required and must name the room; each phase writes journal; a failure at RECREATE restores the compose backup, recreates the old container, and reports measured values; failure after old-container deletion reports "rollback to old container impossible"; health gate ignores a matching line older than start epoch; docker mux bytes are decoded into text; `go2rtc.verify` detects `url: "<name>"`, missing credentials and non-existence; DELETE succeeded / PUT failed is compensated and reported; the shared lock blocks a concurrent watchdog pass.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: Wire P3 UI** — Cameras page: Review diff → typed confirm → Apply → SSE live log; every log line redacted; the confirmation states ALL rooms go briefly deaf.

- [ ] **Step 4: Live Apply on ONE room only with the user's explicit instruction in that moment** (standing rule). Verify after: `/health` 200, five sessions, fresh 12.50 chunks/s window, ~6400 B/s producer, play_audio ratio, feed rms, `config_drift` read-only and clean; audit entry written.

### Task 9: Settings page (compose/.env, dry-run → Apply)

**Files:** Create `settings.py`; modify `static/index.html`; `tests/test_settings.py`.

**Interfaces:**
- Produces: redacted field list from compose/.env (values masked, `set/unset`), `POST /api/settings/preview` returning a staged diff, `POST /api/settings/apply` reusing Task 8's Apply with typed confirmation. Fields in scope: gateway service-wide knobs (LLM/TTS backends, watchdog, Telegram toggles, wake defaults) plus camera v1 fields; secrets never displayed, written only via `env_merge`.

- [ ] **Step 1: Tests** — preview does not write; a secret field shown as `set` survives Apply unchanged; a non-secret edit appears in the diff; typed confirmation enforced; `config_drift` never triggers a camera write.

- [ ] **Step 2: Run** — expected failures; implement; run green.

### Task 10: Cutover gate, parity check, recommendation

**Files:** Create `scripts/parity_check.py`, `docs/cutover.md`; tests under `tests/test_parity.py`.

**Interfaces:**
- Produces: `parity_check.py --live` (compares dashboard endpoints/UI needles against the parity matrix) and `docs/cutover.md` with the P5 evidence bar: parity green, suites green, one real Apply verified with 12.50 chunks/s + a live wake, several days dashboard-only without regressions.

- [ ] **Step 1: Tests** — a missing nav token or endpoint fails parity; `--live` exits non-zero on a failed read; no secret in output.

- [ ] **Step 2: Run** — expected failures; implement; run green.

- [ ] **Step 3: When the bar is met, RECOMMEND cutover to the user in writing.** Do not remove old UIs or start P6 until the user explicitly approves. P6 gets its own spec and plan.

## Execution ledger

Progress/verification results go to `.superpowers/dd/2026-11-25-unified-dashboard/progress.md` (git CLI is unavailable in this sandbox; no commits are possible here). Do not push to GitHub.
