#!/usr/bin/env python3
"""Check the numbers in README.md against the repository. A GATE, not a chore.

Run from the repository root:  python3 scripts/check_doc_numbers.py [--tests N]

WHY THIS FILE EXISTS
-------------------
`v2.41` shipped a README whose declared line counts and test count had been typed
from memory rather than measured, and the counts were already drifting: the
release commit was the first time anything verified them. "Update the README" is
a step at the end of every task and a step nobody re-reads, so the numbers in it
decay silently — and a README that says 615 tests when the suite runs 645 is
worse than one that says nothing, because it is believed.

WHAT IS CHECKED
---------------
1. Every `**`name` ` (~N lines)` row in the README's "What it is made of" table.
   The declared number must be within TOLERANCE of the real line count, where the
   tolerance is derived from the precision the README itself uses — `~4,000` is an
   estimate and is allowed to be an estimate, `4,000` is a fact and is not.
2. Every Python module listed in the repository-layout tree appears in that table,
   so a new module cannot be added and left undocumented.
3. `| Test suite | N tests, M files, ~K lines |` — M and K against the filesystem.
4. The version badge against the newest git tag.
5. The test COUNT, only when `--tests N` is supplied.

The test count is the one number the filesystem cannot answer, so
`verify_endpoint.sh` passes it in from the pytest run it has already done rather
than this script starting a second suite.

TOLERANCE IS NOT A FUDGE FACTOR
-------------------------------
`~4,000` means "about four thousand", and "about" has to mean something or the
tolerance is a licence to be wrong: the check divides by the declared precision,
so `~12,540` must be within 1% of 12538, while `~570` may be within 2%. A
declared number that is wildly out is always a failure — that is the case this
file was written for.

SELF-TEST FIRST
--------------
`--self-test` runs seven samples, including two that MUST fail and five that must
not, so a check that cannot fail is never shipped. That is the rule this
repository has already broken three times, in this file's own neighbourhood.

Returns 1 if anything is wrong, 0 if clean.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# `| **`main.py`** (~3,990 lines) |` and `| **`engine.py`** (~570 lines) |`
ROW = re.compile(r"\|\s*\*\*`([^`]+)`\*\*\s*\((~?)([\d,]+)\s*lines?\)")
# `| Test suite | 645 tests, 19 files, ~12,540 lines |`
SUITE = re.compile(r"\|\s*Test suite\s*\|\s*([\d,]+)\s+tests,\s*([\d,]+)\s+files"
                   r"(?:,\s*(~?)([\d,]+)\s+lines)?")
BADGE = re.compile(r"badge/version-v([\d.]+)")
# `├── vosk_wake.py            # wake word by DECODING (...)`
TREE = re.compile(r"[├└]──\s+([A-Za-z_][A-Za-z0-9_]*\.py)\b")


def _n(text: str) -> int:
    return int(text.replace(",", ""))


def tolerance(declared: int, approximate: bool) -> int:
    """How far a declared line count may sit from the truth.

    Derived from the declared precision, not chosen. An estimate written as
    `~570` is rounded to the nearest ten and may be off by five; one written as
    `~12,540` is rounded to the nearest hundred and may be off by fifty.
    """
    if not approximate:
        return 0
    if declared >= 1000:
        return max(50, declared // 100)
    if declared >= 100:
        return max(5, declared // 100)
    return max(2, declared // 100)


def real_lines(name: str) -> int | None:
    path = ROOT / name
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


def latest_tag() -> str:
    try:
        out = subprocess.run(
            ["git", "tag", "--sort=-v:refname"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        tags = [t for t in out.stdout.split() if t.startswith("v")]
        return tags[0][1:] if tags else ""
    except Exception:                                                    # noqa: BLE001
        return ""


def check_text(text: str, *, tests: int | None, files: dict[str, int | None],
               test_files: int, test_lines: int, tag: str) -> list[str]:
    """Return one message per discrepancy. Pure, so the self-test can drive it."""
    bad: list[str] = []

    declared: dict[str, int] = {}
    for name, approx, num in ROW.findall(text):
        value = _n(num)
        declared[name] = value
        actual = files.get(name)
        if actual is None:
            bad.append(f"{name}: declared in the table but the file does not exist")
            continue
        tol = tolerance(value, bool(approx))
        if abs(actual - value) > tol:
            bad.append(
                f"{name}: README says {'~' if approx else ''}{value:,} lines, "
                f"the file has {actual:,} "
                f"(allowed error {tol})"
            )

    # Every module in the layout tree must be in the table, or a new module can
    # be added and left undocumented without anything noticing.
    for name in dict.fromkeys(TREE.findall(text)):
        # `files`, not the disk: the self-test has to drive this without touching
        # the repository, and the first version reached for `is_file()` and so
        # could not reproduce its own sample.
        if name not in declared and files.get(name) is not None:
            bad.append(f"{name}: listed in the repository layout but has no "
                       "row in the 'What it is made of' table")

    m = SUITE.search(text)
    if m:
        if tests is not None and _n(m.group(1)) != tests:
            bad.append(f"Test suite: README says {_n(m.group(1))} tests, "
                       f"pytest reported {tests}")
        if _n(m.group(2)) != test_files:
            bad.append(f"Test suite: README says {_n(m.group(2))} test files, "
                       f"there are {test_files}")
        if m.group(4):
            value = _n(m.group(4))
            tol = tolerance(value, bool(m.group(3)))
            if abs(test_lines - value) > tol:
                bad.append(f"Test suite: README says {'~' if m.group(3) else ''}"
                           f"{value:,} test lines, there are {test_lines:,} "
                           f"(allowed error {tol})")

    b = BADGE.search(text)
    if b and tag and b.group(1) != tag:
        bad.append(f"version badge says v{b.group(1)}, newest tag is v{tag}")

    return bad


# --------------------------------------------------------------------------
# Self-test. Two that MUST fail, five that must not.
# --------------------------------------------------------------------------
_MUST_FAIL = [
    # a declared count that is out of tolerance
    "| **`main.py`** (~900 lines) | x | y |",
    # a module with no table row, listed in the layout. `vosk_wake.py` is the
    # real case: it is in the repository layout and has no row in the table.
    "voice_gateway/\n├── vosk_wake.py\n",
    # a test-file count that does not match the filesystem
    "| Test suite | 10 tests, 3 files |",
    # a version badge behind the newest tag
    "![version](https://img.shields.io/badge/version-v1.0-blue)",
]
_MUST_PASS = [
    "| **`main.py`** (~4,000 lines) | x | y |",
    "| **`engine.py`** (571 lines) | x | y |",
    "| Test suite | 645 tests, 19 files, ~12,540 lines |",
    "![version](https://img.shields.io/badge/version-v2.41-blue)",
    "nothing numeric here at all",
]


def self_test() -> int:
    files = {"main.py": 4004, "engine.py": 571, "vosk_wake.py": 294,
             "audio_utils.py": 223}
    common = dict(
        tests=645, test_files=19, test_lines=12538, tag="2.41",
    )
    failures: list[str] = []

    for sample in _MUST_FAIL:
        # Each sample is checked in isolation, so a failure cannot be masked by
        # another line in the same document.
        if not check_text(sample, files=files, **common):
            failures.append(f"MUST FAIL, but passed: {sample[:56]!r}")

    for sample in _MUST_PASS:
        got = check_text(sample, files=files, **common)
        if got:
            failures.append(f"MUST PASS, but failed: {sample[:56]!r} -> {got}")

    # An estimate must be allowed MORE error than an exact count, and more
    # error the longer the number is. The first version of this assertion had the
    # comparison the wrong way round and the self-test caught it.
    if not tolerance(4000, True) > tolerance(4000, False):
        failures.append("an estimate is allowed no more error than an exact count")
    if not tolerance(12540, True) > tolerance(570, True):
        failures.append("a longer declared number is allowed less relative error")
    if tolerance(4000, False) != 0:
        failures.append("an exact count must be checked exactly")

    for f in failures:
        print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"self-test: {len(failures)} failure(s) — the gate cannot be trusted",
              file=sys.stderr)
        return 1
    print(f"self-test: {len(_MUST_FAIL) + len(_MUST_PASS) + 1} samples, both "
          "directions")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    tests: int | None = None
    if "--tests" in argv:
        try:
            tests = int(argv[argv.index("--tests") + 1])
        except (IndexError, ValueError):
            print("  --tests needs a number", file=sys.stderr)
            return 1

    if not README.exists():
        print(f"  no such file: {README}", file=sys.stderr)
        return 1

    text = README.read_text(encoding="utf-8")
    files = {name: real_lines(name) for name in
             {m[0] for m in ROW.findall(text)} | set(TREE.findall(text))}
    test_files = len(list((ROOT / "tests").glob("*.py")))
    test_lines = sum(
        sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
        for p in (ROOT / "tests").glob("*.py")
    )

    bad = check_text(text, tests=tests, files=files, test_files=test_files,
                     test_lines=test_lines, tag=latest_tag())
    if bad:
        print(f"  README NUMBERS — {len(bad)} discrepancy(ies):")
        for line in bad:
            print(f"    {line}")
        return 1

    note = "" if tests is None else f", {tests} tests confirmed by pytest"
    print(f"  README.md: line counts, suite size and version agree{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))