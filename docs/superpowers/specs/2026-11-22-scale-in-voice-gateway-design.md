# Kitchen scale (ESP32, 192.168.22.123) in the voice gateway — step 1

Date: 2026-11-22. Status: approach approved by the user ("делай по Подход A").

## Intent and why

The kitchen has an ESP32 scale with an on-device MCP server (`POST /mcp`,
firmware v1.4.2, tools `get_weight` / `tare` / `set_unit`, verified live
22.11.2026). The user wants two voice commands now:

- «сколько весит?» → read the current weight aloud;
- «обнули весы» → tare the scale, with verification.

Step 2 (explicitly out of scope here): recipe-guided cooking where the system
names each ingredient, watches the weight and auto-tares before the next one.

Why L1, not L2: this is a number read and one guarded physical action. The L1
router answers in ~0.1–0.5 s and verifies side effects; L2 costs 45–120 s of
LLM steps and has a long documented history of inventing device names and
reasons. A scale read must never go through a model that can hallucinate.

## Goals

1. Deterministic, sub-second spoken answer to weight questions.
2. `tare` only when the platform is empty, then verified by a fresh read.
3. Honest failure: unreachable scale → explicit refusal sentence, no escalation
   to L2 (which has no scale tools and would invent an answer).
4. Code structured so the same client can be reused by the worker in step 2.

## Non-goals / anti-goals

- No automatic speaking when the reading stabilises (the scale sits at zero
  most of the time; without a policy this becomes constant chatter).
- No recipe state, no ingredient tracking, no `set_unit` voice command yet.
- No new dependency in the router image (aiohttp is already there).
- No changes to `main.py`, `voice_mcp.py` or the gateway image in this step.

## Constraints and measured facts

- Live scale: `initialize` returns `esp32-scale` / protocol `2025-06-18`;
  `tools/list` gives `get_weight` (empty schema), `tare` (empty schema),
  `set_unit` (unit 0..3). `tools/call get_weight` returns JSON-RPC content text
  with `value`, `grams`, `unit`, `stable`, `zero`, `tare`, `sigma_mg`, `cal`,
  `tare_offset`, `uptime`. `POST /mcp` is a stateless single-message endpoint —
  no session handshake, no SSE, no accept header needed.
- The scale's own tool description says: check with `get_weight` that the
  platform is empty before calling `tare`.
- jev-router code is baked into its image; deploy = rebuild + clone-recreate
  through the docker socket (same procedure as 16.11.2026), not `docker restart`.
- `/route` answers SSE, not JSON.
- Host shell is a sandbox with only docker.sock; published/host-network services
  are reached at the host IP (192.168.22.102).

## Approaches considered

- **A (chosen): dedicated `scale` route in L1.** New pure HTTP client module +
  resolver phrases + early interception in `_handle` before the classifier +
  a dedicated execution branch. Fast, deterministic, unit-testable, no LLM.
- **B: worker (L2) MCP tools.** More flexible for step 2, but reading a number
  would cost 45–120 s and can produce invented reasons. Rejected for step 1;
  the client is written importable so B can reuse it later.
- **C: extra REST endpoint in the gateway.** Duplicates routing and adds a hop.
  Rejected.

## Decisions (builder calls, recorded so they can be checked)

- Speech always in grams regardless of the scale's display unit; round to the
  nearest whole gram. Spelled-out number with correct noun agreement
  (грамм / грамма / граммов), same last-two-digits rule the router already uses
  for час/минута in `_datetime_phrase`.
- Tare guard: refuse if `abs(grams) > 5`; otherwise call `tare`, re-read and
  require `zero == true`; log the verdict. Only `get_weight`/`tare` are used —
  `set_unit` is not exposed in step 1.
- Interception happens right after STT repair and before `classifier.classify`,
  so «обнули весы» cannot be rewritten by the easy_action bare-verb fallback
  into a light command. Route event is emitted with `route="scale"`,
  `confidence=1.0`, reason `scale_weight` / `scale_tare`.
- Client uses aiohttp with a short total timeout; env `SCALE_MCP_URL` overrides
  the default `http://192.168.22.123/mcp`. No retries in step 1 (a failed POST
  is reported, not silently repeated).
- Tests: pure resolver/format tests + a stubbed MCP transport (no real device)
  + both-direction checks (break the guard, confirm the test fails).

## What is left to the builder

Nothing unresolved: implement approach A, test, rebuild/recreate jev-router,
verify live with a direct `/route` probe (read the `data:` lines), and record
the field numbers in AGENTS.md.
