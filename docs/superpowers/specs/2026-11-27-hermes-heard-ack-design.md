# Hermes-bound questions: immediate "heard, thinking" acknowledgement — design

**Date:** 27.11.2026
**Approved by:** user ("делай")

## Intent / why

When a question is escalated to Hermes, the answer takes 10-17 s. The user must
not be left in silence: they need to know the question was heard and that the
system is thinking. The `stt_suspect` route shipped earlier this day suppressed
the generic ack entirely, making this gap concrete.

## Decided how

- `CascadeBackend` (the shared path for camera, ESP32 and Telegram) owns the
  phrase, because it already knows `persona` and `honorific` and because all
  deterministic gateway phrases live there.
- New `ACK_HEARD = "Вопрос услышан. Секунду, думаю."`, `DOBBI_ACK_HEARD =
  "Домовой услышал{h}. Секунду думаю."` and `_ack_for_question(persona,
  honorific)`.
- On the SSE `route` event: `route == "expert"` puts the heard-ack immediately
  and cancels any delayed `ack_later`; `complex_logic` keeps the old
  `ACK_COMPLEX`. No reason-specific exceptions — `stt_suspect` is included.
- `put_ack()` writes straight to the queue, so the ack never sets `content`
  and cannot suppress the apology if the brain subsequently fails.
- The suspect route still emits its "I may have misheard" cue at the END;
  the two phrases have different jobs and both remain.

## Non-goals

- No change to the router, the wake path, the follow-up window, or any camera
  setting.
- No ack for fast `easy_query`/`easy_action` routes — they answer in <1 s.
- No new timer: the ack must be first in the queue on the route event.

## Size

Small change: two constants, one method, one branch, tests in
`tests/test_cascade_backend.py`. Gateway restart only — `backends.py` is
bind-mounted; the router image is untouched.
