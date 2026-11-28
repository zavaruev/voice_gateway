# Suspect STT: misheard phrase handling — design

**Date:** 27.11.2026
**Approved by:** user ("пока кажется что все правильно придумано, делай")

## Intent / why

Whisper periodically mistranscribes real speech into fluent but wrong Russian
(field case 10:15 MSK 27.11: «как звали Пушкина?» → «так звали Пушкина», no
«?»). The acoustic gates see real speech (`raw_rms=2813`, `peak=26608`), so they
pass. The classifier returns `complex_logic conf=0.0 reason=ambiguous(...)`;
`is_question_shape()` sees no leading question word and no «?», so the v2.87
upgrade-to-Hermes guard and the last-resort fallback both skip. The tool agent
burns 120 s, the gateway times out, and the room says only an apology.

The system has no state meaning "I probably misheard". This feature adds it.

## Goals

- Detect the SHAPE of an utterance with no recognisable intent (not specific
  misspellings), at L1, on the deterministic path.
- Never send such a turn to L2: go straight to Hermes with a note that the
  transcript may be inaccurate.
- Tell the user, in speech, that the phrase was likely misheard and ask them
  to say the correct version; open the existing follow-up window so the
  correction is accepted as a normal turn with that room's history.
- Keep all device/action routing, wake gating, echo guards and persona rules
  unchanged.

## Non-goals / anti-goals

- No per-word repair tables (no «так звали» → «как звали»).
- No second Whisper pass, no wake decoder as second opinion (measured bad).
- No acoustic level threshold (the failing turn had loud real speech).
- No new learning from corrections in this iteration; `CAMERA_FOLLOWUP=none`
  remains an explicit opt-out of the window.

## Decided how

- Pure predicate in `services/jev-router/resolver.py`:
  `is_unrecognised_utterance(text)` — no question shape, no knowledge
  question, no device noun, no action-verb family.
- Router (`app.py`): for `route == complex_logic` and that predicate, set
  `route="expert"`, `reason="stt_suspect"`, `knowledge=True`.
- `chat_proxy.stream_chat(..., stt_suspect=False)`: when true, append a
  system-prompt note that the transcript may be inaccurate; answer by
  meaning, ask to clarify only if meaning is unclear. Hermes-first order
  unchanged.
- After the Hermes stream, router emits a final deterministic `sentence`
  event `_STT_SUSPECT_ACK`:
  «Похоже, я неверно расслышала. Если это не то — повторите, пожалуйста,
  правильную фразу.»
  `dobbi_style()` whitelist maps it to the masculine
  «Домовой неверно расслышал{h}. Повторите, пожалуйста, правильную фразу.»
- The cue is last on purpose: the follow-up window opens after the whole
  reply, not in the middle. It contains «повторите, пожалуйста», which
  `_is_question_reply()` already recognises, so `CAMERA_FOLLOWUP=question`
  opens a 30 s window with `_followup_is_question=True` (peak gate skipped).
- `CascadeBackend` suppresses its generic «Секунду, занимаюсь…» ack for
  `reason=stt_suspect`; other routes unchanged.

## Constraints from the codebase

- Tests are TDD, RED watched before GREEN; checks in both directions for
  wiring (a test that calls the function itself proves nothing about the
  call site).
- `services/jev-router` is baked into the image: rebuild + clone-recreate.
  `backends.py` is bind-mounted: restart suffices.
- Containers log UTC, host is MSK: report times in MSK (UTC+3).
- No state-changing camera actions without an explicit instruction — this
  change touches only router/gateway code.
