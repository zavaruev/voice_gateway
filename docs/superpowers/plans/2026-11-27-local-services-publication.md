# Publish three local services as standalone GitHub projects — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `identity-service`, `coqui-tts-server` and `silero-tts-server` as three public, self-contained GitHub repos with English docs/comments and no house data.

**Architecture:** Stage sanitized copies of the three local services under `voice_gateway/.publish/`, translate every comment/docstring to English while keeping Cyrillic string literals intact, run two self-tested gates (leak scan + comment-language scan), then create the repos through the GitHub MCP and push the staged text files. Nothing in the live service directories is edited or deleted.

**Tech Stack:** Node 22 (gates, local static file server), GitHub MCP via Code Mode `tools.github.*`, Markdown, Docker Compose, Python sources as data.

**Spec:** `docs/superpowers/specs/2026-11-27-local-services-publication-design.md`

## Global Constraints

- Three repos only: `identity-service`, `coqui-tts-server`, `silero-tts-server`. No `whisper-stt-stack`.
- Owner `zavaruev`, public (`private: false`), branch `main`, no `autoInit`.
- GitHub MCP is callable **only inside `execute`**; the shell here has no `git`/`docker` CLI, but has Node and a docker socket.
- Every published file is UTF-8 text. No `data/`, models, mp3, `*.log`, backups, pycache.
- Leak rule: any real `192.168.22.x` address, token prefix, chat id or real profile name found in a staged file → **STOP that repo, report file and line**, no silent redaction.
- Language rule: all comments/docstrings in English; Cyrillic **string literals** stay (they are data).
- Test fixtures: real profile names replaced by `person1/person2/person3`; honorific data stays Russian.
- License files copied from `voice_gateway/LICENSE` and `COMMERCIAL-LICENSE.md` (repo name substituted in the latter).

## Review Focus

- A real IP/token/name planted in a staged file must stop the push — pinned by Task 1's self-test on both the positive and negative sample.
- A Cyrillic comment must be flagged while a Cyrillic string literal must not — pinned by Task 1's language-gate self-test.
- `identity-service` compose must not contain `/mnt/media/...` paths or `env_file:` of live secrets — pinned by Task 2's grep check.
- `silero_tts/Dockerfile` is `FROM coqui-tts:latest`: the coqui README must build exactly that tag — pinned by Task 4's cross-check.
- A Cyrillic comment hidden in a large helper (`num2ru.py`, `yo_restore.py`) must be caught — pinned by Task 1's scan of the full staged file list, not a sample.

---

### Task 1: Staging layout and two self-tested gates

**Files:**
- Create: `.publish/gates/leak_gate.js`, `.publish/gates/lang_gate.js`
- Test: `.publish/gates/selftest.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `node .publish/gates/leak_gate.js <file>...` exit 0 clean / 1 leak; `node .publish/gates/lang_gate.js <file>...` exit 0 clean / 1 non-English comment.

- [ ] **Step 1: Write the failing self-test**

Create `.publish/gates/selftest.sh` asserting: `leak_gate.js` exits 1 on a fixture containing `192.168.22.102`, an `sk-` token, a 9-digit chat id and `marina`, and exits 0 on a fixture with placeholders only; `lang_gate.js` exits 1 on `# комментарий` and on a triple-quoted Russian docstring, exits 0 on `# comment` and on `logger.info("Привет")`.

- [ ] **Step 2: Run it and watch it fail**

Run: `bash .publish/gates/selftest.sh`
Expected: FAIL — gate scripts do not exist.

- [ ] **Step 3: Implement the gates**

`leak_gate.js`: regex scan over the given files for the four literal classes above, print `file:line: class` and `process.exit(1)` on any hit. `lang_gate.js`: minimal Python-aware state machine tracking triple-quoted strings and `#` comments outside strings; flag Cyrillic only in comments/docstrings, never in ordinary string literals.

- [ ] **Step 4: Run the self-test and a deliberate break**

Run: `bash .publish/gates/selftest.sh`
Expected: PASS. Then temporarily break each gate (remove one pattern / skip docstring branch), rerun, and confirm the corresponding assertion FAILS; restore.

- [ ] **Step 5: Record the passing baseline**

```sh
node .publish/gates/leak_gate.js .publish/gates/selftest.sh   # expected: 1 (fixture strings)
```
Expected: the gate does fire on its own self-test file — confirm it is not accidentally excluded from scans.

### Task 2: Stage `identity-service`

**Files:**
- Create: `.publish/identity-service/**` (copy of source files listed in the spec)
- Modify: `.publish/identity-service/tests/*.py` (neutral fixtures), `.publish/identity-service/docker-compose.yml` (portable)

**Interfaces:**
- Consumes: `.publish/gates/*` from Task 1.
- Produces: a directory of publishable text files, all gates clean.

- [ ] **Step 1: Copy exactly the spec's file set**

Use `cp` for: `app.py`, `profiles.py`, `engines.py`, `photos.py`, `curator.py`, `scripts/topup_faces.py`, `static/index.html`, all `tests/*.py`, `pytest.ini`, `requirements.txt`, `Dockerfile`, `.gitignore`. Do NOT copy `data/`, `identity.tar`, `__pycache__`, `.pytest_cache`.

- [ ] **Step 2: Replace real fixtures**

In `.publish/identity-service/tests/*.py`, replace `alexander`/`marina`/`vladislava` with `person1`/`person2`/`person3` everywhere, keeping the Russian honorific values. Run the leak gate after; Expected: no hits for those names.

- [ ] **Step 3: Make the compose portable and write new files**

Rewrite `.publish/identity-service/docker-compose.yml` per spec: relative `./data`, `./speakers`, `./known_faces`, `./face_pending`; remove both `env_file:` entries; `${SPEAKER_ID_URL:-http://localhost:8001}`, `${FACE_API_URL:-http://localhost:8090}`, `${GATEWAY_URL:-http://localhost:18792}`, `${IDENTITY_ROOMS:-room1,room2}`. Write English `README.md` (prerequisites, env table, `/health`, run), `LICENSE`, `COMMERCIAL-LICENSE.md` (name substituted), `.env.example`, `.gitignore`.

- [ ] **Step 4: Verify the staged tree**

Run: `grep -R -n -e '/mnt/media' -e 'env_file' .publish/identity-service/docker-compose.yml; node .publish/gates/leak_gate.js $(find .publish/identity-service -type f); node .publish/gates/lang_gate.js $(find .publish/identity-service -type f)`
Expected: no grep output, both gates exit 0.

### Task 3: Stage `coqui-tts-server`

**Files:**
- Create: `.publish/coqui-tts-server/**`

**Interfaces:**
- Consumes: gates from Task 1.
- Produces: self-contained text-only coqui service; the exact image tag `coqui-tts:latest`.

- [ ] **Step 1: Copy the source set and translate comments**

Copy `server.py`, `warmup.py`, `yo_restore.py`, `num2ru.py`, `Dockerfile`, `requirements.txt`. Translate every Russian/Chinese comment or docstring in the four Python files **and in the Dockerfile** to English; keep all string literals (including the `ё`-restore dictionaries and num2ru word forms) verbatim.

- [ ] **Step 2: Write portable compose and docs**

`docker-compose.yml`: one service, `image: coqui-tts:latest`, `build: .`, `PORT=${PORT:-5051}`, `CUDA_VISIBLE_DEVICES=0`, `device_ids: ['0']`, `./models` bind mount, healthcheck. English `README.md` must document `/v1/audio/speech`, `/v1/audio/speech/stream`, `X-Sample-Rate`, `COQUI_TOS_AGREED=1`, first-start XTTS download, and `docker build -t coqui-tts:latest .`. Add `LICENSE`, `COMMERCIAL-LICENSE.md`, `.env.example`, `.gitignore`.

- [ ] **Step 3: Verify**

Run: `node .publish/gates/leak_gate.js $(find .publish/coqui-tts-server -type f); node .publish/gates/lang_gate.js $(find .publish/coqui-tts-server -type f); grep -c 'coqui-tts:latest' .publish/coqui-tts-server/README.md`
Expected: gates exit 0, grep count ≥ 1.

### Task 4: Stage `silero-tts-server`

**Files:**
- Create: `.publish/silero-tts-server/**`

**Interfaces:**
- Consumes: the English-translated `yo_restore.py`/`num2ru.py` from Task 3 (copy them, do not retranslate); gates.
- Produces: self-contained silero service whose Dockerfile base resolves.

- [ ] **Step 1: Copy and translate**

Copy `server.py`, `Dockerfile`, plus the two translated helpers from `.publish/coqui-tts-server/`. Translate Russian comments/docstrings in `server.py` **and the Dockerfile** to English; keep string literals.

- [ ] **Step 2: Write portable compose and docs**

`docker-compose.yml`: one service, `SILERO_VOICE=${SILERO_VOICE:-baya}`, `PORT=${PORT:-5053}`, `device_ids: ['0']`, healthcheck, `start_period: 120s`. English `README.md`: build coqui first (`docker build -t coqui-tts:latest .` from the coqui repo) because the Dockerfile is `FROM coqui-tts:latest`; 147 MB model downloaded at build; warm-up before port opens; speaker list; `speed` accepted and ignored. Add `LICENSE`, `COMMERCIAL-LICENSE.md`, `.env.example`, `.gitignore`.

- [ ] **Step 3: Cross-check the image tag and verify**

Run: `grep -Fq 'FROM coqui-tts:latest' .publish/silero-tts-server/Dockerfile && grep -Fq 'coqui-tts:latest' .publish/coqui-tts-server/README.md; node .publish/gates/lang_gate.js $(find .publish/silero-tts-server -type f); node .publish/gates/leak_gate.js $(find .publish/silero-tts-server -type f)`
Expected: exit 0; a mismatch here is a hard stop.

### Task 5: Publish and verify

**Files:**
- No local files modified; GitHub state only.

**Interfaces:**
- Consumes: `.publish/<repo>/**` from Tasks 2–4.
- Produces: three public repos with verified contents.

- [ ] **Step 1: Serve staged files locally**

Start a temporary Node static server rooted at `.publish/` on `127.0.0.1` in the background. Code Mode `execute` has no filesystem access, so the publish script must `fetch('http://127.0.0.1:PORT/<repo>/<file>')` to obtain each file's UTF-8 text before calling the GitHub tools.

- [ ] **Step 2: Create the three repos**

In one `execute` call: `tools.github.create_repository({name, description, private:false, autoInit:false})` for each of the three names. Record the responses; abort if any is not public.

- [ ] **Step 3: Push each repo in one commit**

In `execute`: for each repo, fetch all staged files and call `tools.github.push_files({owner:"zavaruev", repo, branch:"main", files:[{path, content}, ...], message:"Initial import from the home lab"})`. If `main` does not exist, retry after a README-only push. Do not mix repos in one commit.

- [ ] **Step 4: Verify remote state**

Call `tools.github.get_file_contents({owner, repo, path:"README.md"})` and one source path per repo; confirm content matches the staged bytes. Call `tools.github.search_repositories({query:"user:zavaruev"})` and confirm all three exist with `private: false`.

- [ ] **Step 5: Stop the local server and report**

Kill the Node server. Report the three `html_url`s, file counts, and gate results. Any repo that failed a gate was never created/pushed.

## Self-review notes

- Spec coverage: 3 repos, no whisper; English docs/comments; no house data; leak stop; neutral fixtures; tag cross-check — all mapped to tasks.
- `writing-plans` requires the executor to choose subagent-driven or native before work begins; that choice is asked for when this plan is handed over.
