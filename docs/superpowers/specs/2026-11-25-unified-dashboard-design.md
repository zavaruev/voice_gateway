# Unified Dashboard — one web entrance for the house

Date: 2026-11-25. Status: approach A chosen by the user; design approved in
conversation; the user delegated the "is it safe to delete the old UI" call to
this agent with the final decision reserved to the user.

## Intent and why

The house has three separate web surfaces: the gateway dashboard on :18792
(ESP devices, sessions, MCP console, firmware/OTA), the identity service on
:18794 (people, voice samples, face photos, camera recording), and nothing at
all for cameras or service-wide settings — camera configuration lives in
`ai-prod/docker-compose.yml` and is edited by the agent by hand.

Two concrete pains: (1) the user delegates camera additions but has no
visibility into what is being configured, and (2) there is no single place to
go. The user asked for a Hermes-style system dashboard: a live overview plus
safe management, in exactly the visual language of the identity service.
Cameras specifically must be editable through a safe config editor with a
staged diff and an explicit Apply, not by invisible edits to compose.

## Goals

1. One new web entrance (Dashboard) that becomes the only place the human
   visits. It serves ported copies of BOTH existing UIs in the identity design
   language, plus new Status, Cameras and Settings pages.
2. Camera configuration is transparent: per-field provenance (source env line,
   go2rtc stream, wake model) and live health numbers, with staged diff →
   Apply → live logs.
3. Apply performs the whole chain itself: anchored compose edit → clone-recreate
   via docker.sock → health gate → go2rtc re-registration → verification,
   with durable journal, global lock, rollback and audit log.
4. Zero changes to the live gateway during phases P0–P4. Old UIs and APIs stay
   alive as fallback until field-verified. Removal is a separate, later,
   explicitly approved step.
5. `main.py` decomposition is a SEPARATE project (P6), additive and
   behaviour-preserving, started only after the dashboard is verified.

## Non-goals / anti-goals

- No merging of gateway/identity backends into the dashboard. They keep their
  APIs; the dashboard proxies them and serves UI.
- No camera-side state changes in v1: no `ifconfig`, `S95majestic restart`,
  reboot, direct `/play_audio`, or camera config write. Only compose edit,
  gateway container recreate and go2rtc stream re-registration.
- No auto-repair from `config_drift` — the standing rule forbids it.
- No plaintext secrets in UI, staged state, diff, errors, logs or audit trail.
- No GitHub push until field confirmation.
- No deletion of old UIs and no `main.py` refactor before verification.

## Constraints and measured facts

- Gateway `:18792`: Basic admin auth; `network_mode: host`; bind-mounted
  per-file sources, `config/`, `logs/`; **no** docker.sock, **no** compose
  mount. Dashboard HTML is `templates/index.html` (Devices / Sessions /
  Control tabs, device CRUD, MCP console, firmware upload). `main.py` is
  ~201 KB; module-level globals are rebound at call time by TTS switching;
  Telegram helpers are injected (no `import main`); CI stages top-level
  `*.py` and `services/`, `templates/`, `tests/`.
- Identity `:18794`: own Basic admin + Bearer read token, fail-closed;
  `identity/static/index.html` is a single file (no framework), design tokens:
  `--blue:#0066cc`, black 44 px nav, 52 px frosted sub-nav
  `blur(20px) saturate(180%)`, 1200 px grid `minmax(320px,1fr)`, 18 px card
  radius, exactly ONE box-shadow (avatars), light/dark with `localStorage`,
  `Cache-Control: no-store`. Room list from `IDENTITY_ROOMS`. Photos are kept
  in `data/photos/<key>/orig/` (enrollment) + crops for display.
- Cameras are **not** a runtime registry. `CAMERA_STREAMS` and every
  `*_<NAME>` variable live in compose (and gitignored `.env.voice_gateway`)
  and are read once at gateway startup. Adding/removing a room requires
  **clone-recreate**, not `docker restart`. go2rtc re-registration uses
  `DELETE /api/streams?src=<name>` and `PUT /api/streams?name=<name>&src=<url>`
  (both as query params — `src` in the body silently no-ops). The URL must be
  verbatim (credentials + `#backchannel=1`); a self-referential
  `url: "<name>"` means permanent `DESCRIBE 404`.
- Config edits must be anchored replacements required to occur **exactly
  once**, write nothing on mismatch, re-parse and verify the result, keep
  timestamped backups. Several past incidents came from sloppy edits.
- Healthy-room verification numbers: `vosk diag` 12.50 chunks/s (the 300 s
  interval and counters reset on restart — a fresh window is required),
  go2rtc producer ~6400 B/s, `play_audio` ratio x0.1–x1.1, `feed rms`
  ~300–1100, `config_drift` clean.
- Docker logs need 8-byte multiplex-header demux; raw bytes have corrupted a
  past "verification". Loguru prints UTC while the host is MSK (3 h skew);
  gate on container start epoch, never on stale text lines.
- `camera_watchdog.sh` runs on cron and can restart the gateway and cameras;
  Apply must coordinate with it via a shared lock.
- Docker socket = host-root equivalent. No docker CLI in the sandbox; use the
  socket API (stop → rename → clone-create with Config.Env/HostConfig → start
  → health gate → delete old only after health).

## Approaches

- **A (chosen): separate Dashboard service.** One HTML file in the identity
  design language + small FastAPI, own port, own Basic auth, proxy to gateway
  and identity with service credentials. Keeps the live audio process
  untouched and gives a clean permission boundary — the same shape as
  Hermes' machine-level dashboard.
- **B: extend gateway dashboard (:18792).** Rejected: gives the live audio
  process self-modifying power and docker.sock; `main.py` is already huge.
- **C: extend identity (:18794).** Rejected: mixes biometrics with host admin
  and puts docker.sock in the smallest service.

## Decisions (builder calls, recorded)

- Port **18795**, bind to host IP only (`192.168.22.102`), not `0.0.0.0`.
- One Basic login for the dashboard, fail-closed with `WWW-Authenticate`;
  per-Apply typed confirmation is the second gate. No separate read/apply
  roles in v1. Audit log for every Apply, mandatory.
- Compose is authoritative for cameras. `.env.voice_gateway` is displayed
  redacted (set/unset, like Hermes API Keys) and edited by merging changed
  keys into the ORIGINAL file on disk — never round-trip displayed state.
- v1 camera edit scope: add/remove a room plus `CAMERA_STREAMS`,
  `GO2RTC_SOURCE_URL_<NAME>`, `CAMERA_PLAY_AUDIO_URL_<NAME>`,
  `WAKE_VOSK_MODEL_<NAME>`, `CAMERA_WEBRTC_<NAME>`. Per-room tuning fields
  are read-only with provenance in v1.
- Whole-gateway restart for a single-room edit is accepted, and the typed
  confirmation must say that ALL rooms go briefly deaf.
- Env values read once at boot are labelled "effective since <start>"; clocks
  are labelled UTC/MSK everywhere.
- Old UIs stay behind a legacy flag until the user, on the agent's explicit
  recommendation, approves their removal.

## Apply state machine (with compensation)

`STAGED → CONFIRMED (typed room/action) → LOCK → BACKUP → EDIT → RECREATE →
HEALTH → STREAM → OBSERVE → DONE`

- **LOCK**: one global Apply lock, also honoured by `camera_watchdog.sh`.
  Durable journal on disk so an exit between EDIT and RECREATE can be
  explicitly resumed or aborted.
- **BACKUP**: compose, `.env`, and exact go2rtc stream JSON (including
  non-existence), mode 0600, outside any UI-served directory.
- **EDIT**: anchored exactly-once replacements; re-parse; verify; on mismatch
  write nothing and stop.
- **RECREATE**: stop → rename → clone-create → start via docker.sock.
- **HEALTH**: `/health` 200 plus expected startup lines filtered by container
  start epoch; stale lines never count.
- **STREAM**: snapshot exact prior config → DELETE → PUT verbatim URL →
  re-read and verify → detect self-referential `url: "<name>"`. Compensate the
  gap between DELETE and PUT. Coordinate with the gateway's own rate-watchdog
  heal (last writer wins; its lossy fallback must never be allowed to replace
  the credentials).
- **OBSERVE**: fresh measurement window with timeout; on timeout report
  "not yet measurable" honestly, never a green checkmark.
- Every phase has a named compensation and a structured outcome: what
  changed, what did NOT roll back, with measured values. If the old container
  was already deleted, say so plainly.

## Pages and feature parity

Status (live overview, auto-refresh), Cameras (provenance + staged diff +
Apply + live logs), Devices/ESP (device/session CRUD, MCP console, firmware/
OTA — proxied from gateway), People (identity profile cards, voice/face/
camera recording — proxied from identity), Settings (compose/.env editor,
dry-run → Apply), Services (links to router, Frigate, go2rtc, face, TTS,
identity, gateway).

The port is validated against a **feature-parity matrix** built from the
actual endpoints and HTML of both services — theme toggle, `no-store`,
Basic+Bearer acceptance, photo crops and quality verdicts, 401 challenge, MCP
console, device CRUD, firmware/OTA ownership checks. Port by matrix, not by
eye.

## Rollout

- **P0** scaffold + Status + Services (read-only, no docker writes).
- **P1** People (read, then write).
- **P2** Devices/ESP (read, then write).
- **P3** Cameras (read-only provenance, then staged diff + Apply).
- **P4** Settings (compose/.env, dry-run → Apply).
- **P5** cutover: old UIs behind legacy flag. Agent recommends when the bar is
  met; USER makes the final deletion decision.
- **P6 (separate project, after verification)**: additive decomposition of
  `main.py`. New modules top-level only, no `import main`, `main.py`
  re-exports every symbol, both-direction tests, CI staging and `__new__`
  session helpers preserved. Full suite + live smoke before any deletion.

P5 evidence bar: parity matrix green, suites green, at least one real camera
Apply verified with 12.50 chunks/s and a live wake, several days of
dashboard-only operation without regressions. The agent states when this is
met; the user decides.

## Tests

Unit: diff engine (exactly-once anchors, redaction, no-write-on-mismatch),
auth and proxy, staged state machine, compensation outcomes, secret handling,
UTC/MSK and start-epoch log filtering. UI: design tokens and page presence.
Integration: docker and go2rtc stubs only — no real destructive ops in tests.
Keep the gateway suite green (1028 passed / 1 skipped / 1 known pre-existing
ring-buffer fail); add a dashboard suite.

## Builder review incorporated

Secret round-trip destruction; redaction of the verbatim RTSP URL in
provenance/diff/errors/SSE logs; durable journal + global lock + watchdog
coordination; go2rtc race (snapshot incl. non-existence, self-reference
detection, compensation between DELETE and PUT); stale-log freshness;
structured rollback outcomes including "old container already deleted";
feature-parity matrix from real endpoints; exact mounts/permissions;
"effective since" labels; clone-recreate deafness in confirmation;
`config_drift` read-only; P6 module/re-export/CI constraints.

## What is left to the builder

Exact env var names, the 18795 port allocation check, service credential
wiring from the gitignored env, and the precise parity test cases are
implementation details settled in the plan.
