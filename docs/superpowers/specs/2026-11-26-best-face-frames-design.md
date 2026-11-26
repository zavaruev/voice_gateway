# Best-face-frames curator — keep the eight best, not the first eight

Date: 2026-11-26. Status: approach A chosen by the user; after measurement the
user chose a cap of **24** instead of 8 and asked to TOP UP all profiles to 24
and keep selecting the best frames continuously.

## Intent and why

Vladislava has 8 face photos in identity because her archived source set had
only 8 frames; Alexander and Marina have 24 because the importer caps at 24.
The code default of the face engine is 8 samples, but the DEPLOYED
`face-recognition` container runs `MAX_SAMPLES_PER_PERSON=24`, and its `/health`
reports alexander 24, marina 23, vladislava 21 embeddings. The earlier "8 is the
model's real capacity" claim was wrong and was retracted to the user before any
build. The user wants every profile topped up to 24 identity photos, with bad
frames replaced by good ones as they arrive, so the gallery always holds the
best 24 currently available.

## Goals

1. One-time top-up of every profile's identity photo library to the best 24,
   sourcing additional frames from `pending/_archive`, `_enrollment` and, for
   Vladislava, quality-checked raw frames in `captures/` recognised as her;
   the face engine is rebuilt from those same originals.
2. On each recognized door arrival, consider the new frame: if it is
   measurably better than the current worst by ≥10 %, replace that frame,
   rebuild the engine set and archive the removed frame.
3. All writes go through the face engine HTTP API; no direct `.npy` writes.
4. Every removal is backed up and audited; nothing silently disappears.

## Non-goals / anti-goals

- No camera state change of any kind (no WiFi bounce, majestic restart, reboot,
  config write, `/play_audio`).
- No change to greeting, Telegram, voice, wake or routing behaviour.
- No change to vote rules (2-frame vote, 0.20 floor, 0.08 ambiguity margin,
  30 s arrival cooldown) or to `over_threshold` (logged only).
- No deletion of biometrics without a backup and an audit record.
- No GitHub push without field confirmation.

## Measured constraints

- `identity/photos.py:141-187`: startup import is one-shot (`.seeded`), sources
  are `pending/_archive/*/<key>`, `_enrollment/<key>`, `<key>`; cap 24; stores a
  display crop and an untouched original plus `manifest.json`.
- `identity/app.py:276-313`: `POST /api/profiles/{key}/face/enroll` recognises,
  crops, saves originals, calls `enroll_face(replace=False)`, prunes rejected.
- `identity/app.py:315-338`: per-photo DELETE rebuilds the whole set from the
  remaining **originals** with `replace=True`; engine-first, local delete second.
- Face engine `server.py:408-412` quality gates (det ≥0.6, face ≥60 px, blur
  ≥20, brightness 30–235), `:427-454` cross-match 0.6, `:466-479` cosine dedupe
  0.98; the code default `MAX_SAMPLES_PER_PERSON = 8` is OVERRIDDEN in the live
  container to 24 (`MAX_SAMPLES_PER_PERSON=24`), and `trim_samples` removes the
  most redundant row when appending past the cap. Do NOT change the engine env
  in this work.
- `face_events._save_pending()` (`face_events.py:506-526`) already runs off the
  arrival path with name, absolute frame path and a fresh conf ≥0.6; it copies
  full frames to `pending/<name>/` for `face-retrain`. Frame verdict jsonl has
  `quality_ok`, `face_size`, `det_score`, `bbox` (`face_capture_daemon.py`).
- Identity's local photo dedupe is by filename only, not content hash.

## Approaches

- **A (chosen): curator in identity.** New identity endpoint accepts a frame,
  scores it, stores original+crop, ranks all originals, rebuilds the engine
  with `replace=True`, archives the removed files. Gateway calls it from the
  existing `_save_pending()` hook. Single source of truth, reuses existing
  enroll/delete semantics, no second quality implementation.
- **B: curate in gateway's `pending/` and re-import.** Rejected: identity's
  `.seeded` marker blocks incremental import, and the logic would be split.
- **C: curate inside the face engine.** Rejected: touches the CUDA service,
  leaves identity's gallery out of sync.

## Decisions (builder calls)

- Keep exactly **24 originals** per person in identity, matching the DEPLOYED
  engine cap; if the engine's dedupe yields fewer embeddings, report the real
  count rather than claiming 24.
- Quality score = `det_score × bbox area` — intrinsic, so it stays comparable
  after every `replace=True` rebuild. Recognition `confidence ≥0.6` is a gate
  for accepting the identity, not part of the ranking. `quality_ok` and the
  capture blur/brightness gates are enforced upstream; no JSONL plumbing into
  identity. Stored per photo in `manifest.json`.
- When an engine rebuild rejects a frame, roll back the local write (including
  HTTP 400) and keep the previous set; never leave an un-enrolled photo behind.
- "One frame per arrival" = the best-scoring of the event's already recognised
  matches, with the hourly gap checked once per event. For the one-time top-up,
  raw `captures/` frames are scored through `/recognize` and only accepted when
  the name matches and conf ≥0.6.
- Replace only when the new score exceeds the current worst by ≥10 %; otherwise
  do nothing. At most one frame per recognized arrival, and at most once per
  hour per person (`FACE_BEST_FRAME_MIN_GAP_S`), to avoid churn.
- Removed originals and crops are copied to `data/photos/<key>/_pruned/<UTC>/`
  (mode 0600) before deletion, with a manifest entry and an audit line.
- One-time cleanup scores every existing original via the face engine API,
  keeps the best 8, rebuilds once per profile, archives the rest.
- `FACE_BEST_FRAMES_ENABLED` gates the arrival hook; it ships off in code and
  is switched on in compose only after the one-time cleanup is verified.

## Flow on arrival

`_save_pending()` → existing copy into `pending/<name>/` (unchanged) → if flag
on and per-person gap elapsed → POST frame to identity `/api/profiles/<name>/face/curate` →
identity: `/recognize` (name + conf + bbox + det_score) → score → store original
+ crop + content SHA1 → append to manifest → rank all originals → if new is not
in top 24, keep files but skip rebuild → else rebuild `replace=True` from the
24 best, archive removed files, audit.

## Tests

Identity: scoring, top-8 selection, 10 % margin, content-SHA1 dedupe, backup
before delete, rebuild from originals, engine failure leaves files intact,
unauthorized call refused. Gateway: arrival hook calls identity once, honours
flag/gap, does not touch camera state. Integration with stubbed face engine.
One-time cleanup is executed live only with the user's explicit approval
(already given for pruning all profiles to 8), verified by photo counts and
`/api/profiles` flags.

## Rollout

1. Unit/integration suite green with stubs.
2. One-time top-up of all three profiles to 24 (Vladislava from 8 by backfilling
   quality-checked recognised frames from `captures/`); verify engine
   `sample_counts`, identity photo counts, and no false recognition on a
   known-camera probe.
3. Enable `FACE_BEST_FRAMES_ENABLED=1` in compose; watch real arrivals for a day.
4. Only then consider any deletion of the old `pending`/archive data (separate
   approval).

## What is left to the builder

Exact env names, the one-time cleanup script location, and the manifest schema
extension are implementation details settled in the plan.
