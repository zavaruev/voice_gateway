# Math routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A math/knowledge utterance is either evaluated by L1's calculator or escalated to Hermes — never dispatched to the tool agent, and never answered with a "misheard" cue.

**Architecture:** `calc.py` gains the two missing Russian constructions; `resolver.py` recognises all arithmetic vocabulary as knowledge under the existing device/home guards; `app.py` sends a failed knowledge question to `expert` even without a question mark.

**Tech Stack:** Python 3.12, pytest; router image only (gateway tests import its source).

**Spec:** `docs/superpowers/specs/2026-11-27-math-routing-design.md`

## Global Constraints

- Device/home questions stay on the home ladder (`_DEVICE_NOUN`, `RE_HOME_QUESTION`).
- `route` event is emitted once; `question_to_expert`/`knowledge` set before it.
- No changes to STT, wake, cameras, audio, or L2.

## Review Focus

- A math question with no `?` → expert/calc, never `complex_logic`.
- A math phrase with a device noun → home path, not expert.
- A bare `%` (e.g. «двадцать процентов») remains `(20/100)`, not a multiplication.
- An unsafe/invalid expression remains refused by `calc.evaluate`.

---

### Task 1: Calculator grammar

**Files:** `services/jev-router/calc.py`, `tests/test_calc.py`

- [ ] Add failing tests: `eval("два процента четырех тысяч") == 80`, `eval("шесть в квадрате") == 36`, `eval("два в кубе") == 8`, `eval("двадцать процентов") == 0.2`, and the existing home-question guard test stays green.
- [ ] Implement `_OPS` postfix power rules and the `% M` → `(N/100)*M` rule after the existing `% от M` rule. Keep the letters guard.
- [ ] Run `pytest tests/test_calc.py -q` — green.
- [ ] Commit (on host; no git CLI in sandbox).

### Task 2: Knowledge predicate

**Files:** `services/jev-router/resolver.py`, `tests/test_router_resolution.py`

- [ ] Add failing tests: `knowledge_question("шесть в квадрате")`, `knowledge_question("два процента четырех тысяч")`, `knowledge_question("умножь три на четыре")` all True; `knowledge_question("сколько процентов заряда пылесоса")` stays False.
- [ ] Broaden the math branch of `RE_KNOWLEDGE` with those vocabulary stems.
- [ ] Run the resolver tests — green.

### Task 3: Routing and full verification

**Files:** `services/jev-router/app.py`, `tests/test_router_resolution.py`

- [ ] Add failing route-level tests using the exact field texts: with the classifier forced to `complex_logic/0.18`, `включи`-free math text yields `route=expert` and `knowledge=True`; the action/download paths are unchanged.
- [ ] Change the redirect condition to `(is_question_shape(text) or knowledge_question(text))`.
- [ ] Run targeted tests, then the full suite in the voice_gateway image (source + tests PUT), expecting only the known ring-buffer failure.
- [ ] Rebuild + health-gated clone-recreate `jev-router`; probe `/route` with both field texts read-only (no device toggle); confirm the SSE route is `calc` or `expert`, not `complex_logic`.
