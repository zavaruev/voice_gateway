# Unified identity service (Phase B) — one profile store over two engines

Date: 2026-11-25. Status: approach A chosen by the user ("Делай"); the last open
question answered ("ну сделай интерфейс, запишем новые образцы голосов") and
the design treated as approved. Builder review is in `## Builder review`.

## Intent and why

Phase A made Добби address people by honorific. To do that it had to know who
was speaking, and the answer was scattered across four places: the voice
gallery (`speakers/*.npy`), the face gallery (`known_faces/*.npy`),
`config/speaker_names.json` + `config/person_honorifics.json` in the gateway,
and a hardcoded `TG_PERSON` chat map in `main.py`. Adding a person or changing
an address currently means editing files in two projects and redeploying.

The user wants one identity API that owns a person's profile — display name,
honorific, Telegram chat ids, and which of the two engines has samples — plus
an interface for adding samples later. Marina and Vladislava have faces and
Telegram ids but no voice samples yet; the user explicitly said samples will
be added through the interface, so an **incomplete profile must be a normal
profile**, not a blocker. Alexander has voice samples available for testing.

## Goals

1. One API that is the source of truth for person → {name, honorific, Telegram
   ids} and for "does this person have voice/face samples".
2. One web interface to create profiles and add voice samples / photos. Writes
   go to the two existing engines through their own APIs; this service never
   writes a `.npy` file directly.
3. The gateway can read identity (person + honorific in one answer) from the
   new API, with an env switch that restores the current behaviour byte-for-byte.
4. Nothing in Phase A's spoken behaviour changes: same prompt wording, same
   wake gates, same deterministic routes, same honorific rules.

## Non-goals / anti-goals

- **No merging the two model processes into one container.** The images have
  conflicting pins (speaker-id: torch cu118; face: `onnxruntime-gpu==1.17.1`
  with `numpy<2` on CUDA 11.8). Validate compatibility before even proposing
  it; it is not part of this work.
- No change to a camera's state (no WiFi bounce, restart, config write, reboot
  or direct `/play_audio`). Enrolling Alexander's already-supplied voice
  samples through the new interface IS authorised; nothing else is.
- No GitHub push until the user confirms field behaviour.
- No deletion of the Phase A config files in this step: they remain the
  read-only fallback seed until the new path is confirmed in the field.
- No voice enrollment for Marina/Vladislava (no samples supplied).
- No change to wake words, Dobbi phrasing, greeting logic, or L1/L2 routing.
- No second voice stack: the interface only manages identity data, it speaks
  nothing.

## Constraints and measured facts (verified 25.11.2026, not assumed)

- **Voice engine** `speaker_id_service`, published :8001, container up 2 days.
  `POST /identify` (multipart `file`) → `{user_id, confidence}`; score floor 0.4
  in the engine, 0.1 in the gateway. `POST /enroll` needs form `user_id`,
  `files` (≤50) and header `X-API-Key`, compared against the `API_KEY` env var
  (`speaker_id_service/.env`, gitignored). Gallery host path:
  `/mnt/media/docker-compose/speaker-id/speaker_id_service/speakers/*.npy`
  (alexander, plus two test profiles). No delete endpoint. Image is
  `python:3.11-slim` + torch cu118.
- **Face engine** `face-recognition`, published :8090, container up 3 days.
  `GET /health` → `known_faces` + `sample_counts`; `/recognize` (multipart
  `file`) → faces with `name`/`confidence`; `/enroll` (form `name`, `files`,
  `mode=individual` default, quality gates + cross-match rejection);
  `GET /enroll/list`; `DELETE /enroll/{name}`. Gallery host path
  `/mnt/media/docker-compose/face-captures/known_faces/*.npy` (alexander,
  marina, vladislava). `server.py` reloads galleries on **latest mtime** via
  `reload_if_changed()` — `face-retrain` writes `.npy` from another process and
  depends on this. Every write must therefore go through the face API, or the
  running container will not see it. Image: CUDA 11.8 + ONNX Runtime 1.17.1.
- **Gateway** runs `network_mode: host` (18792 REST, 18793 MCP) with
  `config/` and `logs/` bind-mounted. `main.py` loads `SPEAKER_NAME_MAP` and
  `HONORIFIC_MAP` once at import; `TG_PERSON` is a literal dict of three chat
  ids; `camera_client._current_honorific()` reads `_last_uid` directly from the
  honorific map; `face_events` calls `FACE_API/health` and `FACE_API/recognize`.
- Host shell is a sandbox with only docker.sock: reach host-network services at
  `192.168.22.102`; new services must be created through the socket API
  (there is no docker CLI here). A rebuild/recreate is needed for env changes;
  a restart is not enough.
- Test material: real recorded utterances in
  `voice_gateway/utterance_samples/*.wav`, plus the already-enrolled
  `speakers/alexander.npy`.

## Approaches considered

- **A (chosen): identity API owns profiles and drives both engines.** A new
  small FastAPI service stores profile metadata in one atomic JSON file, calls
  the two engines' own APIs for enrollment and recognition, and returns a
  single identity payload (person + honorific) to the gateway. The gateway's
  old paths stay as the fallback behind an empty `IDENTITY_URL`, exactly as
  `TTS_BACKEND=""` preserves the legacy envs. Real single source of truth,
  no model merging, no camera changes.
- **B: thin façade only** (proxy endpoints, metadata still in gateway config
  files). Less code, but it does not unify management and the interface would
  still need to edit two projects' files. Rejected.
- **C: merge voice + face models into one process.** Ruled out by the pinned
  dependency conflict above and by different GPU allocations; not attempted.
- **D: only unify the gateway's config into one JSON.** No UI, no shared
  enrollment, no identity API — does not meet the request. Rejected.

## Decisions (builder calls, recorded so they can be checked)

- **New service** `/mnt/media/docker-compose/ai-prod/identity/`, plain
  `python:3.11-slim` + FastAPI + uvicorn + aiohttp + python-multipart — no GPU
  libraries. `network_mode: host`, port **18794** (verify free before deploy).
  Same host-IP address convention as the other services. No new LLM, no TTS.
- **Storage**: `data/profiles.json`, bind-mounted, written atomically
  (tempfile in the same dir + `os.replace`) under an asyncio lock. Schema
  `{"version": 1, "profiles": {key: {display_name, honorific,
  telegram_chat_ids, tts_voice, notes, updated}}}`. Sample presence is **read
  from the galleries** via two read-only mounts, never duplicated as a stored
  boolean that could drift. Unknown engine users (test profiles) are not
  imported as people; an unknown uid maps to no honorific, never a guess.
- **Write path**: voice enroll → `POST :8001/enroll` with `X-API-Key` (key read
  from the same gitignored `.env`, never logged); face enroll →
  `POST :8090/enroll`. No direct `.npy` writes. A failed engine call is
  returned as an error with status and reason — no silent success, because a
  guard that cannot report failure is worse than none.
- **Read path**: `POST /api/identify` proxies `:8001/identify` and joins the uid
  to profile {person, display_name, honorific}; `POST /api/recognize` proxies
  `:8090/recognize` and joins each name. `GET /api/profiles` returns metadata
  merged with gallery presence and engine rosters.
- **Auth**: two independent secrets. `IDENTITY_TOKEN` (Bearer) for gateway
  reads/proxies; `IDENTITY_ADMIN_USER`/`IDENTITY_ADMIN_PASSWORD` (Basic) for
  the UI and all writes. Empty secret = fail closed (401/503), never an open
  admin page. Never reuse the gateway admin Basic, `MCP_TOKEN` or
  `FACE_GREETING_TOKEN`.
- **Gateway integration**: `IDENTITY_URL` env, empty by default. When set:
  `fetch_speaker_id` asks identity and stores both person and honorific
  (`_current_honorific()` prefers the identity answer, falls back to the local
  map); `face_events` takes its roster and `/recognize` from identity;
  Telegram's chat→person map is built from identity profiles at startup.
  Empty `IDENTITY_URL` keeps every current call site untouched.
- **Migration**: a one-off local script seeds `profiles.json` from
  `speaker_names.json`, `person_honorifics.json`, the three confirmed Telegram
  ids and the live engine rosters; it prints the diff and writes only after
  the user-approved values are confirmed. Existing config files are left in
  place as fallback until field confirmation.
- **Interface**: one plain HTML page served by the identity service, Basic
  protected. Lists people with voice/face status, Telegram ids and honorific;
  forms to create a profile, edit metadata, add multiple voice files or photos
  per person. Shows engine status codes and errors verbatim; speaks nothing;
  touches no camera.
- **Tests**: unit tests with stubbed engine HTTP (no network, no GPU) for
  mapping, auth fail-closed, atomic write, and the empty-env fallback; each
  wiring test verified in both directions (delete the call, watch the test
  fail). Full in-image gateway suite plus the identity service's own suite.
- **Deploy**: create the container through docker.sock, then recreate the
  gateway with `IDENTITY_URL` (env change needs recreate, not restart).
  Verify read-only first: `/health`, `/api/profiles`, identity `/api/identify`
  on Alexander's existing utterance wav, and one live voice turn with no
  physical lamp probe. No GitHub push.

## Builder review — questions resolved

An independent builder raised six ambiguities. Five were settled as builder
calls; the sixth was answered by the user.

1. **Both speaker-ID call sites switch.** The camera session's own
   `_fetch_speaker_id` (camera_client.py) and the ESP32/Telegram one
   (`main.fetch_speaker_id`) both ask identity when `IDENTITY_URL` is set.
   Both keep returning a **str engine uid** so every existing caller and test
   stays valid; the person/honorific join is stashed in the identity client's
   cached snapshot and read by `_current_honorific()` / the Telegram sender.
   Switching only `main.py` would have left the primary (camera) path on the
   old map — caught by the review.
2. **Outage fallback.** With `IDENTITY_URL` set but identity unavailable or
   erroring: log one warning naming the failure, then use the current Phase A
   path for that turn (direct engine URL + local map). Never silence the room
   and never let a stale file override a UI edit *while identity is up*. An
   engine uid with no profile maps to no honorific (""), never a guess.
3. **Voice samples — user's answer.** Voice `/enroll` REPLACES the centroid
   (`alexander.npy`), there is no append API, and the only audio on disk is
   729 room recordings containing wake words/TV/commands. The user chose:
   **build the interface; record clean samples later and add them through it.**
   Therefore: do NOT enroll anything during this work, do not touch
   `alexander.npy`, and do not test enrollment with `utterance_samples/*.wav`.
   Verification uses the existing gallery read-only (feed a real utterance to
   `/api/identify`, expect `alexander` → `сэр`). The UI must state plainly
   that voice enrollment replaces the existing sample set. Face enroll stays
   append (`replace=false`).
4. **Live edits.** Profile changes made in the UI take effect without
   recreating the gateway: the identity client keeps a short-TTL (30 s) cached
   `/api/profiles` snapshot, refreshed on demand, rather than a startup-only
   snapshot. This is the point of the feature — a startup snapshot would keep
   the redeploy the user wants to remove.
5. **Face call sites.** Identity owns the greeting roster (`/api/profiles` →
   `face_greeting.warm`) and `/recognize`. `frigate_events.py` keeps calling
   `FACE_API/detect` directly: detection is not identity, and changing it
   would risk the snapshot-injection path for no benefit.
6. **Values in existing logic.** `person` stays the engine key (`alexander`,
   `marina`, `vladislava`) so `face_greeting`'s fallback map, time-bucket cache
   keys and Telegram captions keep working unchanged. `display_name` is
   decoration only. Unknown gallery users (`testuser`, `testcurl`) are not
   imported as profiles — absence from `profiles.json` is enough, no
   deny-list; they resolve to no honorific.

## What is left to the builder

Nothing unresolved. Implement approach A with the six resolved answers, test
both directions, deploy behind the empty-env fallback, migrate the three
profiles, verify read-only with Alexander's existing gallery, and record the
measured numbers in `AGENTS.md`. Clean voice samples are recorded by the user
later and added through the new interface.
