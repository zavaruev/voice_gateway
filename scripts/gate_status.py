#!/usr/bin/env python3
"""Run every gate, print one table, and record when each gate last FIRED.

Run from the repository root:  python3 scripts/gate_status.py [--json]

WHY
---
`verify_endpoint.sh` had grown to seven steps and four gates, each printing its
own block. That is fine at four and unreadable at eight, and the failure mode is
not cosmetic: **a gate whose result is scrolled off the screen is a gate that
stops being run**, and then it protects nothing while still being in the repo.

Two things this adds, and the second is the point:

1. ONE TABLE, so every gate's verdict is on one screen and none is missed.
2. **AUTOMATIC HISTORY.** "When did this gate last catch something?" cannot be a
   hand-maintained column — that is a number in a document, and the reason
   `check_doc_numbers.py` exists in the first place. Every run appends its real
   outcome to `logs/gates.jsonl`, and the "last fired" column is read back from
   that. It cannot drift.

THE FAILURE THIS IS BUILT AGAINST
---------------------------------
`scripts/check_*.py` existing is not the same as it running. This script treats
**an unregistered gate as a failure**: any file matching `check_*.py` that is not
in the registry below is reported, because a gate nobody wired into the gate runner
is indistinguishable from no gate at all. That is the same shape as the bug in
`check_lang.py`'s own self-test, where the first sample passed on the code it was
meant to refute.

A gate that has never fired is not a defect — `check_env_wiring` has never fired
and that is correct. It IS worth seeing, which is why the column exists.

Returns 1 if any gate fails or any gate is unregistered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "logs" / "gates.jsonl"

# gate name -> (what it catches, how to run it)
GATES: dict[str, tuple[str, list[str]]] = {
    "check_lang": (
        "Russian prose outside quoted data in code and docs",
        ["python3", "scripts/check_lang.py"],
    ),
    "check_doc_headings": (
        "a heading spliced to a copy of itself in AGENTS.md",
        ["python3", "scripts/check_doc_headings.py"],
    ),
    "check_env_wiring": (
        "an env var set in compose that no code path reads",
        ["python3", "scripts/check_env_wiring.py"],
    ),
    "check_doc_numbers": (
        "README counts drifted from the repository",
        # The test COUNT cannot be derived from the filesystem, so it needs the
        # number pytest reported. `verify_endpoint.sh` exports it as
        # VOICE_GATEWAY_TEST_COUNT; without it this runs count-blind and says so
        # rather than reporting a clean pass it did not earn. The first version
        # omitted the argument entirely, so a README claiming 600 tests against a
        # real 645 passed this runner and failed only in the pipeline — a check
        # that runs differently from how it is used is not a check.
        ["python3", "scripts/check_doc_numbers.py"],
    ),
}

# Appended when the environment provides the number; see the note above.
TEST_COUNT_ENV = "VOICE_GATEWAY_TEST_COUNT"

DATE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def discover() -> list[str]:
    """Every `check_*.py` in scripts/, so an unwired gate is visible."""
    return sorted(
        p.stem for p in (ROOT / "scripts").glob("check_*.py")
    )


def run(name: str, cmd: list[str]) -> tuple[int, str]:
    """Exit code and the first line of its finding, if any."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    except Exception as exc:                                            # noqa: BLE001
        return 2, f"{type(exc).__name__}: {exc}"
    out = (p.stdout or "") + (p.stderr or "")
    finding = ""
    for line in out.splitlines():
        s = line.strip()
        if not s or s.startswith(("self-test:", "SELF-TEST")):
            continue
        if re.match(r"^[✅✓]", s) or s.startswith(("README.md:", "voice_gateway:")):
            continue
        if s.startswith(("check_lang:", "check_doc_headings:", "AGENTS.md:")):
            continue
        if re.search(r"(FAIL|discrepanc|absent|MISSING|found|never read|agree|clean|"
                     r"reachable|splices)", s, re.I) and not s.startswith("  "):
            finding = s
            break
    return p.returncode, finding


def record(results: dict[str, dict]) -> None:
    """Append this run. The history is what the "last fired" column reads."""
    try:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", encoding="utf-8") as fh:
            for name, r in sorted(results.items()):
                fh.write(json.dumps({
                    "ts": r["ts"], "gate": name,
                    "code": r["code"], "finding": r["finding"][:200],
                }, ensure_ascii=False) + "\n")
    except Exception:                                                  # noqa: BLE001
        # The history is a convenience. A read-only log directory must not stop
        # the gates from running — that would be trading a real check for a
        # record of it.
        print("  (could not append to logs/gates.jsonl — continuing)")


def last_fired(gate: str) -> str:
    """The most recent run of this gate that returned non-zero.

    "unrecorded" and "never" are DIFFERENT answers and are kept apart on
    purpose. This history began on 17.10.2026, and `check_doc_numbers` had
    already demonstrably fired before it existed — the column saying "never"
    would be a false statement about a gate that is known to work. So before the
    history starts, the answer is "unrecorded", and nothing is backfilled: a
    hand-written first row would be exactly the kind of invented fact this
    repository has been corrected for repeatedly.
    """
    if not HISTORY.exists():
        return "unrecorded"
    try:
        rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:                                                  # noqa: BLE001
        return "unreadable"
    if not rows:
        return "unrecorded"
    bad = [r for r in rows if r.get("gate") == gate and r.get("code")]
    if not bad:
        return "never"
    return bad[-1]["ts"][:16]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    count = os.environ.get(TEST_COUNT_ENV, "").strip()
    results: dict[str, dict] = {}
    for name, (_what, cmd) in sorted(GATES.items()):
        argv = list(cmd)
        note = ""
        if name == "check_doc_numbers":
            if count:
                argv += ["--tests", count]
            else:
                # Never silently the weaker check.
                note = " (test count NOT checked — set %s)" % TEST_COUNT_ENV
        code, finding = run(name, argv)
        results[name] = {"code": code, "finding": finding + note, "ts": now}

    unregistered = [g for g in discover() if g not in GATES]

    if not args.no_record:
        record(results)

    if args.json:
        print(json.dumps(
            {"results": results, "unregistered": unregistered},
            ensure_ascii=False, indent=2))
    else:
        w = max(len(g) for g in GATES) + 2
        print()
        print(f"  {'GATE'.ljust(w)}{'NOW':<6}{'LAST FIRED':<18}WHAT IT CATCHES")
        print(f"  {'-' * (w + 6 + 18 + 46)}")
        for name in sorted(GATES):
            r = results[name]
            mark = "ok" if r["code"] == 0 else "FAIL"
            print(f"  {name.ljust(w)}{mark:<6}{last_fired(name):<18}{GATES[name][0]}")
            if r["code"]:
                print(f"  {'':<{w}}{'':<6}{'':<18}  -> {r['finding']}")
            elif r["finding"] and r["finding"] not in GATES[name][0]:
                # Shown on PASS too. The note that a check ran in a weaker mode is
                # only useful while everything is green — that is exactly when
                # nobody would otherwise notice it.
                print(f"  {'':<{w}}{'':<6}{'':<18}  -> {r['finding']}")
        if unregistered:
            print()
            print(f"  UNREGISTERED — a gate that exists but is never run:")
            for g in unregistered:
                print(f"    {g}")
        print()

    failed = [n for n, r in results.items() if r["code"]]
    return 1 if (failed or unregistered) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))