# Best-face-frames curator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Top every identity profile up to the best 24 face frames and keep replacing the worst with measurably better frames as they are recognized at the door.

**Architecture:** New `identity/curator.py` owns the intrinsic quality score (`det_score × bbox area`), content-SHA1 dedupe and top-24 selection. A new Basic-authenticated identity endpoint adds a frame, rebuilds the face engine from the 24 best originals with `replace=True`, and archives removed files with an audit record. The gateway calls it from the existing arrival hook; a one-time script backfills Vladislava from quality-checked raw captures.

**Tech Stack:** Python 3.11, FastAPI, aiohttp/httpx, Pillow (already in identity), pytest. No new GPU dependencies. Engine's deployed cap stays `MAX_SAMPLES_PER_PERSON=24` — do not change it.

**Spec:** `docs/superpowers/specs/2026-11-26-best-face-frames-design.md`

## Global Constraints

- Cap is **24** (deployed engine cap), not the code default 8. Do not recreate `face-recognition` or change its env.
- Score = `det_score × (bbox width × height)`, intrinsic and comparable across rebuilds. Recognition `confidence ≥ 0.6` is an acceptance gate, not part of the score.
- Accept one frame per recognized arrival, at most once per hour per person (`FACE_BEST_FRAME_MIN_GAP_S=3600`), only if it beats the current worst by ≥10 % once the gallery is full.
- Engine write path only: `POST face:8090/enroll` with `replace=False/True`; never write `.npy` directly.
- A failed rebuild (HTTP 400/502) rolls back the local write; no un-enrolled photo remains.
- Removed files are copied to `data/photos/<key>/_pruned/<UTC>/` (mode 0600) with manifest and audit entries before deletion.
- No camera state change; no change to greeting/Telegram/voice/wake/routing; no GitHub push.
- Sandbox has no `python3`/`git` CLI; tests run inside containers via the docker socket API; no commits.

## Review Focus

- Engine cap mismatch (code 8 vs deployed 24) ⇒ `test_live_engine_cap_is_24` in Task 5 (live `/health` + container env read).
- Score computed from confidence would drift after every rebuild ⇒ `test_score_is_intrinsic_det_score_times_area` in Task 1.
- A 400/502 rebuild leaves an un-enrolled local photo ⇒ `test_engine_error_rolls_back_the_local_write` in Task 2.
- Content duplicates under new filenames accumulate locally ⇒ `test_content_sha1_dedupe` in Task 1 and Task 2.
- A crash or second concurrent arrival deletes a frame twice ⇒ `test_removal_is_archived_once` in Task 2.

---

### Task 1: Curator primitives in identity

**Files:** Create `identity/curator.py`, `identity/tests/test_curator.py`.

**Interfaces:**
- Consumes: nothing.
- Produces: `def frame_score(det_score: float, bbox: list) -> float`; `def content_sha1(path: str) -> str`; `def select_top(entries: list, cap: int = 24) -> tuple[list, list]` where each entry has `score`, `sha1`, `name`; `CAP = 24`, `MIN_IMPROVEMENT = 0.10`.

- [ ] **Step 1: Tests** — hand-derived values: `frame_score(0.7, [0,0,100,200]) == 14000`; a lower det/area pair scores less; content SHA1 of identical bytes matches under different filenames; `select_top` of 26 entries returns exactly 24 keeps and 2 drops with the lowest scores; equality on score does not replace.
- [ ] **Step 2: Run** `pytest tests/test_curator.py -v` in the identity image; expected failures on missing module.
- [ ] **Step 3: Implement** pure functions; no I/O in `frame_score` or `select_top`.
- [ ] **Step 4: Run** — expected pass.
- [ ] **Step 5: Record in ledger** (no git CLI).

### Task 2: Identity curate endpoint

**Files:** Modify `identity/app.py` (new route + imports), `identity/profiles.py` (manifest score/sha1 helpers if needed), `identity/Dockerfile` (use `COPY *.py ./` + `COPY scripts/ ./scripts/`), `identity/engines.py` (reuse `enroll_face` / `delete_face`).

**Interfaces:**
- Consumes: `curator.frame_score`, `curator.content_sha1`, `curator.select_top`, `profiles_mod` store, `photos_mod.save_photos`, `client.enroll_face(name, paths, replace=True)`.
- Produces: `POST /api/profiles/{key}/face/curate` (Basic) with multipart `file`; returns `{status, kept, removed, engine_samples, score}`.

- [ ] **Step 1: Tests** (`identity/tests/test_curate_route.py`) with a stubbed face client:
  - `test_curate_adds_a_better_frame_and_archives_the_worst` — 24 existing entries + a better new one → 24 kept, 1 removed, one file in `_pruned/`, engine called with `replace=True` and 24 originals.
  - `test_engine_error_rolls_back_the_local_write` — stub raises 400/502 → local new files gone, manifest unchanged.
  - `test_content_sha1_dedupe` — identical bytes under a new filename → no duplicate added.
  - `test_removal_is_archived_once` — repeated curate with the same set does not re-archive.
  - `test_curate_requires_admin_basic` — 401 with challenge.
- [ ] **Step 2: Run** — expected failures on missing route.
- [ ] **Step 3: Implement** the route: recognise the frame (`/recognize`, name must equal key, conf ≥0.6), score it, save original + crop, merge manifest with score/sha1, run `select_top`, if new not in top-24 and gallery full → return without engine call, else rebuild with `replace=True`, archive dropped files, audit. Reuse the existing DELETE rebuild path; on `EngineError` restore the manifest and delete only the files just written.
- [ ] **Step 4: Run** — expected pass; re-read the edited route (hand-kept edit trap).
- [ ] **Step 5: Rebuild identity image via docker socket API; do not deploy to users yet.**

### Task 3: Gateway arrival hook

**Files:** Modify `voice_gateway/face_events.py` (`_save_pending` at ~506-526, flags near 54-66), `voice_gateway/docker-compose.yml` (`FACE_BEST_FRAMES_ENABLED`), `voice_gateway/tests/test_face_events.py`.

**Interfaces:**
- Consumes: identity client recogniser already used by `_recognize_frame`; `settings.face_best_frames_enabled`, `FACE_BEST_FRAME_MIN_GAP_S`.
- Produces: `async def maybe_curate_best(name, matches, settings)`; calls identity curate once per event with the best-scoring match; no camera calls.

- [ ] **Step 1: Tests** — flag off → no POST; flag on + gap not elapsed → no POST; flag on + matches → POST with the best `frame_path` and correct `key`; a POST failure only logs; no `camera`/`ssh`/`play_audio` call in the test's patched session.
- [ ] **Step 2: Run** — expected failures.
- [ ] **Step 3: Implement**, inserted after the existing `shutil.copy2` into `pending/<name>/`; the hourly gap is per person and checked once per event; never import identity app code into the gateway.
- [ ] **Step 4: Run** — expected pass; `bash -n`-equivalent syntax check for changed module via `python -m py_compile`.

### Task 4: One-time top-up script

**Files:** Create `identity/scripts/topup_faces.py`, `identity/tests/test_topup.py`.

**Interfaces:**
- Consumes: `identity/curator.py`, identity `/recognize`, `/api/profiles/{key}/face/curate`.
- Produces: CLI `python scripts/topup_faces.py [--dry-run]`; for each profile, gather archive/enrollment/raw `captures/` frames with `quality_ok`, recognise, accept only name-matching conf ≥0.6, rank, and curate until 24 or candidates run out; prints per-person counts.

- [ ] **Step 1: Tests with stubbed filesystem and HTTP** — `test_topup_fills_vladislava_from_raw_captures` (8 existing + 16 accepted raw = 24), `test_topup_does_not_remove_a_better_existing_frame`, `test_dry_run_makes_no_writes`.
- [ ] **Step 2: Run** — expected failures.
- [ ] **Step 3: Implement**; default dry-run, explicit `--apply` required for writes.
- [ ] **Step 4: Run** — expected pass; re-read the script.
- [ ] **Step 5: Live DRY RUN only** — record how many candidates exist for each person; no writes.

### Task 5: Live verification and enabling

**Files:** Modify `ai-prod/docker-compose.yml` (`FACE_BEST_FRAMES_ENABLED`), ledger.

- [ ] **Step 1: Test** `test_live_engine_cap_is_24` — read `/health` and container env; assert cap 24 and sample_counts ≥ old values; no writes.
- [ ] **Step 2: Run the one-time top-up with `--apply`** — the user explicitly authorised topping all profiles up to 24; verify identity photo counts (24/24/24 if candidates exist), engine sample_counts, zero false recognitions on a known-camera probe, audit entries and `_pruned/` backups.
- [ ] **Step 3: Enable the arrival flag in compose**, health-gated clone-recreate of the gateway (env change — restart is not enough), and watch real arrivals; no camera state change.
- [ ] **Step 4: Record** all measured counts and any person that could not reach 24 in the ledger; report honestly.

## Notes

- No commits possible (no git CLI); progress ledger is
  `.superpowers/sdd/2026-11-25-unified-dashboard/progress.md` extended with this plan's entries.
- Tests cannot run on the host; use the existing docker-socket test runner pattern
  (identity image for identity tests, dashboard/identity image with mounts for gateway tests).
