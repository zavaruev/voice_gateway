# Publish local services as standalone GitHub projects — design

Date: 2026-11-27

## Intent and why

The voice_gateway installer (same session) can only bring up a stack that other
people can actually fetch. Three services in this house are local-only code with
no public repository and are actually used by the live stack: `ai-prod/identity`,
`coqui/`, `silero_tts/`. The user's decision: publish them as **three separate
public repositories** under `zavaruev`, each a self-contained project with its
own README, licence, env example and a portable compose file.

`whisper-gpu/` was considered and dropped: no whisper container runs on this
host at all. The live STT endpoint is `192.168.22.111:8000` on a different host
and serves `koekaverna/faster-whisper-podlodka-turbo`; the local compose boots a
different model (`medium`) and is unused deployment leftovers. Publishing an
unused compose that does not serve the model the gateway asks for would be
misleading, so STT stays an external prerequisite of the installer.

This is a publication/packaging task, not a rewrite. Behaviour stays byte-for-byte
where possible; only host-specific paths, secrets and credentials are changed to
placeholders.

## Goals

- Three public repos: `identity-service`, `coqui-tts-server`, `silero-tts-server`.
- Each repo builds and starts on a clean machine with only documented env vars.
- No live secrets, real camera credentials, profile data, model weights or
  binary assets in any repo.
- Dual AGPL-3.0-or-later / commercial licence, consistent with `voice_gateway`.

## Non-goals

- No code refactor, no shared npm/pip package, no new features. The duplicate
  `yo_restore.py`/`num2ru.py` in the two TTS repos is intentional: each project
  must build from a single clone.
- No publication of `frigate/config`, `homeassistant/config`, `hermes/config.yaml`,
  `OmniRoute/data` or `face-captures`. Those contain live keys (the Hermes config
  genuinely does) and personal device data; they are deployment state, not
  projects. `face-captures` is already a private repo on GitHub despite the
  README calling it public — out of scope here.
- No local git history to preserve: none of the three directories has a `.git`.
  Publishing is done with the GitHub MCP tools (`create_repository` +
  `push_files`), which carry text file contents only.

## Constraints

- GitHub account `zavaruev`; MCP is authenticated and can create repos and push
  text files; it cannot push binary files. Everything published is UTF-8 text.
- `identity/docker-compose.yml` currently hardcodes `/mnt/media/...` and two
  `env_file:` entries pointing at live secret files. The published compose must
  read all secrets from one `.env` with empty defaults (fail closed).
- `coqui`/`silero` compose files pin GPU `device_ids: ['1']`/`['2']` because this
  host has two spare GPUs. The published compose uses one GPU (`device_ids:
  ['0']`) and documents the two-instance variant in the README.
- `num2ru.py` in both TTS repos uses `pymorphy3==2.0.6`; requirements already pin
  it. `silero_tts/Dockerfile` is `FROM coqui-tts:latest` — the README must say
  to build the coqui image first.

## Approaches considered

1. **One monorepo `voice-stack`** — easier cross-service fixes, but contradicts
   the user's "separate projects" and makes each service unpublishable on its own.
2. **Three repos (chosen)** — each service is independently cloneable, has its own
   README/licence and can be installed separately by the Hermes installer. The
   cost is duplicated helper files, which is acceptable and deliberate.
3. **Publish nothing, document local paths** — rejected: the installer cannot
   fetch local paths.

## Decided how

### Repo 1: `identity-service`

Source: `/mnt/media/docker-compose/ai-prod/identity`.

Include: `app.py`, `profiles.py`, `engines.py`, `photos.py`, `curator.py`,
`scripts/topup_faces.py`, `static/index.html`, `tests/*`, `pytest.ini`,
`requirements.txt`, `Dockerfile`, `.gitignore`.
Write new/portable: `README.md`, `LICENSE`, `COMMERCIAL-LICENSE.md`,
`.env.example`, `docker-compose.yml`.
Exclude: `data/` (names, Telegram ids, photos), `identity.tar`, `__pycache__/`,
`.pytest_cache/`.

The compose keeps the service's env names but replaces absolute host mounts with
relative ones (`./data`, `./speakers:ro`, `./known_faces:ro`, `./face_pending:ro`)
and drops both `env_file:` entries; `IDENTITY_*` tokens/user/password come from
`.env`. `SPEAKER_ID_URL`/`FACE_API_URL`/`GATEWAY_URL` become `${...}` with the
host-network defaults documented in the README. `IDENTITY_ROOMS` keeps its
default list but the README says it is deployment-specific.

### Repo 2: `coqui-tts-server`

Source: `/mnt/media/docker-compose/coqui`.

Include: `server.py`, `warmup.py`, `yo_restore.py`, `num2ru.py`, `Dockerfile`,
`requirements.txt`. Write new/portable: `README.md`, `LICENSE`,
`COMMERCIAL-LICENSE.md`, `.gitignore`, `.env.example`, `docker-compose.yml`.
Exclude: `models/` (1.7 GB), `voices_preview/` (mp3 binaries), `build.log`,
`__pycache__/`.

README documents the OpenAI-compatible `/v1/audio/speech` and stream endpoint,
`X-Sample-Rate`, the COQUI TOS flag, the first-start XTTS download (~1.7 GB), and
the `LD_LIBRARY_PATH` quirk. Compose: single service, `PORT=${PORT:-5051}`,
`device_ids: ['0']`, `./models` bind mount, healthcheck.

### Repo 3: `silero-tts-server`

Source: `/mnt/media/docker-compose/silero_tts`.

Include: `server.py`, `yo_restore.py`, `num2ru.py`, `Dockerfile`.
Write new/portable: `README.md`, `LICENSE`, `COMMERCIAL-LICENSE.md`,
`.gitignore`, `.env.example`, `docker-compose.yml`.
Exclude: `__pycache__/`.

README documents: build coqui first (base image), the 147 MB model downloaded
at **build** time, warm-up before the port opens, the accepted speaker list,
`PORT`/`SILERO_VOICE` env, and that `speed` is accepted and ignored.

### Publication mechanics

- `create_repository(name, description, private: false, autoInit: false)` then
  one `push_files(owner:"zavaruev", repo, branch:"main", files:[...], message)`
  per repo. `autoInit: false` avoids a placeholder README commit. If `main` does
  not exist after repo creation, create it with the initial `README.md` commit
  through `push_files` and verify with `get_file_contents` before continuing.
- Licence text is copied verbatim from `voice_gateway/LICENSE`; the commercial
  licence text is adapted by substituting the repo name. Both are public already.
- Verify after every push: `get_file_contents` on the README and one source file,
  and a repo search to confirm `private: false`.
- No local files are deleted or moved. The three source directories stay untouched.

## Decisions from the builder check

- **Leak rule (user):** audit every file BEFORE any push. If a live camera IP,
  token, chat id or profile name is found in sources/tests/UI: **STOP that repo
  and report the exact file and line**. No silent redaction, no auto-exclusion.
  A repo is published only after a clean audit.
- **Deployment data (user):** README and `.env.example` use **placeholders only**
  — no real `192.168.22.x` addresses, no real room/profile names or chat ids,
  not even as examples. `identity-service` tests are an explicit case: the real
  profile names in their fixtures are replaced by neutral `person1/person2/person3`
  consistently; production behaviour and the Russian honorific data are untouched.
- **Language (user):** all published READMEs/docs AND all comments/docstrings in
  published code are in English. Russian string literals stay — they are data
  (the yo-restore dictionary, num2ru word forms, log messages spoken/heard by
  users), exactly the repo's "prose is English, Cyrillic is data" rule. In
  particular `yo_restore.py`, `num2ru.py`, `coqui/server.py`, `warmup.py`,
  `silero/server.py` and the compose files currently have Russian comments that
  must be translated in the published copies without changing behaviour.
- **My call — prerequisites:** `identity-service` documents speaker-id, face API
  and gateway as required external services. It keeps `network_mode: host`
  (Linux) and exposes the three URLs as env overrides. No stub/degraded mode.
- **My call — installer contract:** branch `main`, one compose file per repo,
  service names and health endpoints identical to the live stack, no separate
  manifest file.
- **My call — image tag:** the coqui README builds exactly `coqui-tts:latest`,
  the tag the silero Dockerfile's `FROM` expects.

## Acceptance checks (before publication)

- A self-testable gate scans every candidate file for live IPs, known token
  prefixes, numeric chat ids and real profile names; it must be verified in BOTH
  directions (a planted leak is found) before it is trusted. Any hit stops that
  repo.
- Only the files listed above are pushed; no `data/`, `models/`, `*.mp3`,
  backups or pycache.
- After each push: `get_file_contents` on README and one source file, plus a repo
  search confirming `private: false`.

## What is left to the builder (my calls, not needing the user)

Repo descriptions, README prose, ordering of the three pushes, and exact
placeholder spellings. These do not change what gets published.

## Approval gate

Nothing is pushed until the user approves this document. After approval:
`writing-plans` → implementation (always test/verify a check before trusting it).
