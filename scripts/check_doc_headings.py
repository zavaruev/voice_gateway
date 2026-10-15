#!/usr/bin/env python3
"""Check AGENTS.md headings for splice damage. A DEMONSTRATION OF WHAT IT CATCHES.

Run from the repository root:  python3 scripts/check_doc_headings.py

Why this file exists. The splice in AGENTS.md (`s[:i] + new + s[i:]`, where
`new` already ends with an anchor) glues a heading to a copy of itself. Five
times on 14–15.10.2026, and **the check I ran afterwards passed while the defect
was present, twice**:

  * the first searched for `**- **` — but the splice reads `PHRASE- **ATTENTION`,
    with the hyphen AFTER the asterisks, so the pattern never matched anything;
  * the second caught `- **` not at the start of a line — but the spliced line
    does start with `- **`, it just contains it TWICE. It caught a different case
    (text glued to the end of the previous line), not this one;
  * the third counted repeated headings by a key taken from the start of the
    line — but the copy is on the SAME line, so the key was unique and no
    repeat was visible.

Hence the rule this file exists to enforce: **a check that has never been seen
failing means nothing.** So `--self-test` below first breaks a line exactly the
way the splice breaks it and confirms the check notices. If it does not, the
script fails — not the document.

Three independent signals, because they catch different things:

  1. `- **` is not at the start of the line — text glued onto the previous line;
  2. `- **` appears more than once in the line — a heading spliced with a copy
     of itself (exactly what the splice does);
  3. two headings with the same beginning — the same heading duplicated whole,
     on different lines.

Returns 1 if it found anything, 0 if clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "AGENTS.md"
MARK = "- **"


def _strip_code(line: str) -> str:
    """Drop fenced blocks and inline code spans.

    Without this the check flags itself: the note about the splice pattern
    quotes `- **` literally, and every such quote looks like a defect. A check
    that always fails is just as useless as no check — it gets switched off.
    """
    return re.sub(r"`[^`]*`", "", line)


def find_glue(lines: list[str]) -> list[tuple[int, str]]:
    """Three independent splice signals, each with its own message."""
    bad: list[tuple[int, str]] = []
    in_fence = False
    for i, raw in enumerate(lines, 1):
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = _strip_code(raw)
        n = line.count(MARK)
        if n == 0:
            continue
        if not line.lstrip().startswith(MARK):
            bad.append((i, f"text glued to the end of a line ({n}x '{MARK}' not at the start)"))
        elif n > 1:
            bad.append((i, f"heading spliced with a copy of itself ({n}x '{MARK}' in the line)"))
    heads = [
        (i, _strip_code(line).lstrip()[len(MARK):].split("**")[0])
        for i, line in enumerate(lines, 1)
        if _strip_code(line).lstrip().startswith(MARK)
    ]
    seen: dict[str, int] = {}
    for num, key in heads:
        key = key[:45]
        if key in seen:
            bad.append((num, f"heading repeated (line {seen[key]}): {key[:45]}"))
        else:
            seen[key] = num
    return bad


def self_test() -> int:
    """Test the CHECK against known-broken samples before trusting it."""
    cases = [
        ("heading spliced with a copy of itself", "- **A** text- **A** text", True),
        ("hyphen inside, not at the start",       "  word- **A** text",      True),
        ("one heading, clean",                    "- **A** text",            False),
        ("ordinary text",                         "  just a line",           False),
        ("two different headings",                "- **A**\n- **B**",        False),
        ("the pattern quoted in a line",          "  signal: `- **` here",   False),
        ("the pattern quoted in a fence",         "  ```\n  - **x**- **x**\n  ```", False),
    ]
    fails = 0
    for name, sample, want in cases:
        got = bool(find_glue(sample.splitlines()))
        verdict = "ok" if got == want else "MISSED"
        if got != want:
            fails += 1
        print(f"  self-test: {name:<38} want {str(want):<5} {verdict}")
    print(f"  self-test: {'all samples passed' if not fails else f'FAILED: {fails}'}")
    return fails


def main() -> int:
    if "--self-test" in sys.argv:
        return 1 if self_test() else 0

    if not DOC.exists():
        print(f"  no such file: {DOC}")
        return 1
    lines = DOC.read_text(encoding="utf-8").splitlines()
    bad = find_glue(lines)
    heads = sum(1 for l in lines if l.lstrip().startswith(MARK))
    if bad:
        print(f"  SPLICE — {len(bad)} in {DOC.name}:")
        for num, why in bad:
            print(f"    line {num}: {why}")
            print(f"      {lines[num - 1][:120]}")
        return 1
    print(f"  {DOC.name}: {heads} headings, no splices")
    return 0


if __name__ == "__main__":
    sys.exit(main())