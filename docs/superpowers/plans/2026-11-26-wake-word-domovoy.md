# Alternate wake word Добби → Домовой Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the spoken alternate wake-word tokens in all five rooms from the «добби» mis-hearing family to the exact token «домовой», without changing persona/voice/backend routing.

**Architecture:** Keep `alt_word="dobbi"` as the internal persona/routing key; change only the token allowlist supplied by `WAKE_ALT_TOKENS_<NAME>` and the matcher/strip logic that consumes it. Remove the gated mis-hearing machinery that existed solely because Добби was mis-decoded.

**Tech Stack:** Python, Vosk (`vosk-model-small-ru-0.22`), pytest, docker socket API for recreation.

**Spec:** `docs/superpowers/specs/2026-11-26-wake-word-domovoy-design.md`

## Global Constraints

- Primary `WAKE_TOKENS` (`компьютер`, `комп`, …) is untouched.
- Internal `alt_word` / `kind` / persona key stays the exact string `"dobbi"`.
- `TTS_VOICE_DOBBI=aidar`, `LLM_BACKEND_DOBBI`, `DOBBI_SYSTEM`, `dobbi_style`, honorifics unchanged.
- Exact token match only — no prefix match, no gated ordinary words.
- Env change requires container **recreate**, not `docker restart`.
- Do not alter camera config, mic gain, thresholds or Vosk model.

## Review Focus

- A live TV phrase containing «домovoy»-like ordinary words must not fire — exact-token test in Task 1.
- Old «добрый/дарби/да/доме/давай без» forms must stop firing — regression test in Task 1.
- A «домовой» turn must still route to the `dobbi` persona, `aidar` voice and the Dobbi ack pool — test in Task 2.
- All five rooms must carry the new env; a missed room keeps the old wake word — env assertion in Task 3.
- `docker restart` does not re-read env; only recreate does — verification step in Task 4.

---

### Task 1: Matcher tokens — exact «домовой», drop the gated family

**Files:**
- Modify: `vosk_wake.py:68-145` (token block), `vosk_wake.py:214-279` (match/kind functions), `vosk_wake.py:485-489` (phonetic-form log)
- Test: `tests/test_two_wake_words.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DOMOVOY_TOKENS = frozenset({"домовой"})`; `transcript_has_wake`, `wake_occurrences`, `wake_kind_in` accept the new token set; `wake_kind_in` still returns `"dobbi"` for an alt-token hit.

- [ ] **Step 1: Write failing tests in `tests/test_two_wake_words.py`**

```python
def test_domovoy_is_an_exact_token_and_old_dobbi_forms_are_gone():
    alt = frozenset(vosk_wake.DOMOVOY_TOKENS)
    assert vosk_wake.transcript_has_wake("домовой включи свет", vosk_wake.WAKE_TOKENS | alt) is True
    for old in ("добрый включи свет", "дарби включи свет", "да включи свет",
                "доме какая погода", "давай без включи свет"):
        assert vosk_wake.transcript_has_wake(old, vosk_wake.WAKE_TOKENS | alt) is False

def test_domovoy_wake_kind_stays_dobbi():
    alt = frozenset(vosk_wake.DOMOVOY_TOKENS)
    assert vosk_wake.wake_kind_in("домовой включи свет", alt, "dobbi") == "dobbi"
    assert vosk_wake.wake_kind_in("домовой компьютер включи свет", alt, "dobbi") == "computer"
```

- [ ] **Step 2: Run the tests, confirm FAIL**

Run: `pytest tests/test_two_wake_words.py -q -k domovoy`
Expected: FAIL — `DOMOVOY_TOKENS` does not exist.

- [ ] **Step 3: Implement the token change in `vosk_wake.py`**

Replace the block at lines 68-145 with a single constant `DOMOVOY_TOKENS = frozenset({"домовой"})` plus a short comment recording the measured reason (rare proper names lose Vosk LM 1-best ranking; «домовой» 12/12 fires, clipping-safe to 13.3 % rail). Delete `DOBBI_TOKENS`, `DOBBI_SOLE_TOKENS`, `DOBBI_NEEDS_VERB`, `_DOBBI_COMMAND_WORDS`, `_DOBBI_QUERY_WORDS`, `_DOBBI_QUERY_FORMS` and `_dobbi_hits`.
In `transcript_has_wake`, `wake_occurrences`, `wake_kind_in` delete the `"добби" in tokens` / `_dobbi_hits` branches and the `not in DOBBI_SOLE_TOKENS` exclusions, leaving plain exact-token matching. In `wake_kind_in` keep `return alt_word or "dobbi"`. Remove the phonetic-Добби log branch at lines 485-489.

- [ ] **Step 4: Re-run the two tests, confirm PASS**

Run: `pytest tests/test_two_wake_words.py -q -k domovoy`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the whole two-wake-word file and repair obsolete Dobbi token expectations**

Run: `pytest tests/test_two_wake_words.py -q`
Expected: tests that asserted the old spellings (`добрый`, `дарби`, `да`, `доме`) as *wake* tokens now fail. Update only their token fixtures to `DOMOVOY_TOKENS`, keeping every `last_kind == "dobbi"` assertion and the persona/backend/voice tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add vosk_wake.py tests/test_two_wake_words.py
git commit -m "wake: switch alternate token from Добби to Домовой"
```

---

### Task 2: Strip tokens and persona continuity

**Files:**
- Modify: `camera_client.py:5053-5060` (`_strip_wake_words`), `camera_client.py:64` (comment)
- Test: `tests/test_camera_client.py`

**Interfaces:**
- Consumes: `vosk_wake.WAKE_TOKENS` and `self._wake_tokens`.
- Produces: `_strip_wake_words` removes the exact «домовой» token on an alt-wake turn; `self._wake_kind == "dobbi"` still selects `DOBBI_ACK_PHRASES`.

- [ ] **Step 1: Write a failing test**

```python
async def test_domovoy_is_stripped_and_keeps_the_dobbi_ack_pool():
    s = _make_session()  # existing helper; set the fields below by hand
    s._wake_tokens = frozenset({"компьютер", "комп", "домовой"})
    s._wake_kind = "dobbi"
    assert s._strip_wake_words("Домовой, включи свет") == "включи свет"
    assert s._strip_wake_words("Домовой Домовой") == ""
```

- [ ] **Step 2: Run it, confirm FAIL**

Run: `pytest tests/test_camera_client.py -q -k domovoy_strip`
Expected: FAIL (the strip path still depends on `DOBBI_STRIP_TOKENS`).

- [ ] **Step 3: Implement**

Delete the `extra = vosk_wake.DOBBI_STRIP_TOKENS if self._wake_kind == "dobbi" else frozenset()` block in `_strip_wake_words`; the token is already present in `self._wake_tokens` from config. Delete `DOBBI_STRIP_TOKENS` from `vosk_wake.py` and fix the comment at `camera_client.py:63-64`.

- [ ] **Step 4: Run it, confirm PASS**

Run: `pytest tests/test_camera_client.py -q -k domovoy_strip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add camera_client.py vosk_wake.py tests/test_camera_client.py
git commit -m "wake: strip Домовой from the command sent to the router"
```

---

### Task 3: Compose env for all five rooms

**Files:**
- Modify: `/mnt/media/docker-compose/ai-prod/docker-compose.yml` — the five `WAKE_ALT_TOKENS_<NAME>` lines (`LIVINGROOM`, `KITCHEN`, `ENTRANCE1`, `CAMERA-CORRIDOR1`, `BALCONY`)

**Interfaces:**
- Consumes: `CameraConfig.alt_wake_tokens` (built from `WAKE_ALT_TOKENS_<NAME>` in `main.py:3867`).
- Produces: every session gets `alt_wake_tokens="домовой"`; `alt_wake_word` stays `"dobbi"`.

- [ ] **Step 1: Add a config test that pins the five rooms**

In `tests/test_two_wake_words.py`, extend the existing env test so it asserts:
`cfg.alt_wake_tokens == "домовой"` and `cfg.alt_wake_word == "dobbi"` for a representative room, and add a plain text assertion that the compose file contains exactly five `WAKE_ALT_TOKENS_` lines all equal to `домовой` and no `WAKE_ALT_TOKENS_*=добби`.

- [ ] **Step 2: Run it, confirm FAIL**

Run: `pytest tests/test_two_wake_words.py -q -k compose_tokens`
Expected: FAIL on the current compose values.

- [ ] **Step 3: Edit compose**

Replace all five `WAKE_ALT_TOKENS_<NAME>=добби|добрый` with `WAKE_ALT_TOKENS_<NAME>=домовой`. Leave every `WAKE_ALT_WORD_<NAME>=dobbi`, `TTS_VOICE_DOBBI`, Vosk model and `CAMERA_STREAMS` line untouched.

- [ ] **Step 4: Run it, confirm PASS**

Run: `pytest tests/test_two_wake_words.py -q -k compose_tokens`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add /mnt/media/docker-compose/ai-prod/docker-compose.yml tests/test_two_wake_words.py
git commit -m "wake: enable Домовой token in all five rooms"
```

---

### Task 4: Full verification and live rollout

**Files:** none (verification only)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a running gateway with the new env and a recorded live verdict.

- [ ] **Step 1: Full suite in the image**

Run the suite in a throwaway `ai-prod-voice_gateway:latest` container with `tests/` and `services/` tar-PUT (the image bakes only a stub test dir) and `pip install -q pytest pytest-asyncio httpx` together.
Expected: no new failures; only the documented pre-existing `test_wake_ring_buffer.py::test_ring_buffer_scores_identically` remains.

- [ ] **Step 2: Recreate (do not restart) `voice_gateway` through the docker socket API**

Stop → rename → create with the patched `Config.Env` → start → `/health` 200 → delete old. A `docker restart` would keep the old `WAKE_ALT_TOKENS` values.

- [ ] **Step 3: Read-only live verification**

After a spoken «Домовой, включи свет», read the log: one `VOSK WAKE trig='домовой …' kind=dobbi`, one answer, no `AUDIO STARVED`, no false wakes on television in `vosk diag`. Check that the log line `wake kind=dobbi pip=… ack=…` still appears (persona retained).

- [ ] **Step 4: Record any live miss**

If a real «домовой» attempt is decoded differently, save its exact hypothesis from `vosk diag` / the vosk dump before changing any token. Do not add a spelling on speculation.
