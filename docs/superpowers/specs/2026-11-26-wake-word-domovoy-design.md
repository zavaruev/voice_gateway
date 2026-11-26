# Short design: alternate wake word «Добби» → «Домовой»

## Intent

All five rooms currently accept a second address word, «Добби». Live audio shows
it is heard far worse than the primary «компьютер». Replace only the **spoken
address token** with «домовой» — a frequent Russian word with the same meaning
(a household spirit / домовой эльф). Keep the existing Добби persona, voice,
reply style, honorifics and separate LLM backend untouched.

## Why this word

Measured in a throwaway container with the production `VoskWakeMatcher` and
`config/vosk-model-small-ru-0.22`, 3 offsets × SNR 20/14/9/5 dB into real room
audio:

| word | fires | clean n-best top | clipping robustness |
|---|---|---|---|
| компьютер | 12/12 | `компьютер` | correct up to 9.5 % rail |
| домовой | 12/12 | `домовой` | correct up to 13.3 % rail |
| добби | 12/12 clean but fails on live speech | `добби` | breaks at 3.3 % → «доббин», 9.4 % → «дать бы» |
| кикимер | 0/12 | `ки ки мир` — target absent from n-best | n/a |

Root cause of the live failures: rare words lose to frequent ones in Vosk's
language model; the matcher sees only 1-best. On a real Dobbi attempt «добби»
was rank 4 behind «да и», «допив», «да ты», «дарби». «Домовой» is frequent
enough to win 1-best and is more clipping-tolerant than «компьютер».

## Goals

- «Домовой» fires as reliably as «компьютер» on live speech.
- The primary «компьютер» path is byte-for-byte unchanged.
- No new false wakes on television (allowlist stays exact, no prefix/gated forms).
- Persona, TTS voice (`aidar`), ack phrases, `dobbi_style`, honorifics and
  `LLM_BACKEND_DOBBI` remain as they are.

## Non-goals

- Do not switch the Vosk model, thresholds, mic gain or camera config.
- Do not change the primary wake word or the TTS backend.
- Do not add speculative spellings for «домовой» — the word decodes exactly in
  every measurement; add variants only from a recorded live miss.

## Approach (chosen) — token swap, keep internal persona key

The internal string `alt_word="dobbi"` is a **persona/routing key** across
`camera_client.py`, `main.py`, `backends.py` and the router. Renaming it would
touch ~19 files and 347 references for no user-visible benefit. Keep it.

Change only what decides *what is heard as the address*:

1. `vosk_wake.py`: `DOBBI_TOKENS` → `DOMOVOY_TOKENS = frozenset({"домовой"})`;
   remove the gated mis-hearing machinery (`DOBBI_SOLE_TOKENS`,
   `DOBBI_NEEDS_VERB`, `DOBBI_QUERY_*`, `_dobbi_hits`, the special cases in
   `transcript_has_wake` / `wake_occurrences` / `wake_kind_in`, the phonetic-form
   log line). Those existed solely because Добби was mis-decoded.
2. `vosk_wake.py`: strip tokens → `{"домовой"}` so the router never sees the
   address word.
3. `docker-compose.yml`: in all five rooms
   `WAKE_ALT_TOKENS_<NAME>=домовой` (`WAKE_ALT_WORD_<NAME>` stays `dobbi`).
4. Tests in `tests/test_two_wake_words.py`: retarget the token tests to
   «домовой», keep the `kind == "dobbi"` assertions, add a regression test that
   the old «добрый/дарби/да» forms no longer fire.

**Why not alternatives:** renaming the persona key is a large, risky diff with
identical behaviour; adding n-best reading would admit the address from lower
alternatives and risks false wakes on television; keeping Добби and adding more
spellings is the whack-a-mole already refuted.

## Deployment / verification

- Gateway code is bind-mounted; env changes require a container **recreate**,
  not a restart. No router/worker rebuild (persona unchanged).
- Verify in-image: targeted wake tests + full suite.
- Live verification (read-only first): `vosk diag` on one room shows
  `kind=dobbi` with a `VOSK WAKE trig='домовой …'`, no false wakes, one answer.
- If a real live miss appears, record its exact hypothesis before adding any
  spelling — do not guess.
