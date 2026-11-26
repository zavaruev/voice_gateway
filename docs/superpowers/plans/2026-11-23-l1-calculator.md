# L1 Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use subagent-driven-development or executing-plans. Steps use checkbox syntax.

**Goal:** spoken arithmetic and percentages are evaluated deterministically in L1.

**Architecture:** new pure module `calc.py` (Russian number words -> expression -> safe AST evaluation), a narrow `resolve_calc` gate in `resolver.py`, and an L0.6 interception in `app.py` after the scale branch.

**Tech Stack:** Python 3.12, `ast`, `math`, FastAPI SSE, pytest.

**Spec:** `docs/superpowers/specs/2026-11-23-l1-calculator-design.md`

## Global Constraints
- No `eval`/`exec`; whitelist AST nodes only.
- Calculator must never steal scale/time/weather/brightness/battery/definition routes.
- Code and comments in English; spoken strings in Russian.
- No new dependencies.

## Review Focus
- A percentage expression with a weight noun must stay on the scale route.
- A definition question («что такое корень из 3») must stay with the expert.
- Non-parseable text must fall through, never produce a wrong number.
- Division by zero / huge powers must not kill the router; return an honest refusal.

---

### Task 1: `calc.py` evaluator and tests
**Files:** Create `services/jev-router/calc.py`, `tests/test_calc.py`.

- [ ] Write failing tests: addition in words, precedence, parentheses, decimals, division, percent of, power, root, invalid expression, division by zero.
- [ ] Implement `evaluate(text) -> float | None` and `sentence(value) -> str`.
- [ ] Run `pytest tests/test_calc.py`, watch RED then GREEN.

### Task 2: resolver gate + app interception
**Files:** Modify `services/jev-router/resolver.py`, `services/jev-router/app.py`; extend `tests/test_calc.py`.

- [ ] Write failing tests for `resolve_calc` guards and the `route=calc` SSE event.
- [ ] Implement `resolve_calc` and the L0.6 branch after scale.
- [ ] Run targeted tests RED then GREEN, then the full suite.

### Task 3: deploy
- [ ] Rebuild + recreate `jev-router` (existing docker.sock runner).
- [ ] Probe `/route` live with the exact field phrasings.
