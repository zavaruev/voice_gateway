#!/usr/bin/env python3
"""Every env var the compose file sets for this service must be read by the code.

Run from the repository root:  python3 scripts/check_env_wiring.py [--compose PATH]

WHY
---
A variable set in `docker-compose.yml` that nothing reads is the quietest failure
in this stack: the container starts, the flag looks enabled in the compose file,
and the setting does nothing. Nothing in the log says so. The project already
records the same class twice — a renamed stream, and a config key the firmware
restored to its default while the file still said otherwise.

WHAT IT CHECKS
--------------
For each variable in the `voice_gateway` service's `environment:` block, the code
must reach it by one of the three ways this codebase actually reads config:

  * `os.getenv("NAME")`  — the usual form;
  * `os.environ.get("NAME", ...)` — used by the loguru sink setup;
  * a DYNAMIC prefix, `f"NAME_{...}"` — the per-room scheme
    (`CAMERA_PAUSE_ENDPOINT_KITCHEN` is read as `f"CAMERA_PAUSE_ENDPOINT_{name.upper()}"`).

The third is why a literal search alone reports twenty-eight false positives, and
why the prefix must be resolved rather than skipped: the whole
`CAMERA_*_<ROOM>` family is built at runtime and a naive check makes this gate
look broken on the first run.

SCOPE IS ONE SERVICE ON PURPOSE
-------------------------------
The compose file also defines `jev-router`, `smolagents-worker` and the rest, and
their variables (`HA_TOKEN`, `QDRANT_URL`, `WORKER_TIMEOUT`) are read by their own
code. Scanning the whole file reports thirty-odd false positives; scanning only
`voice_gateway` currently reports zero.

A GATE THAT HAS NEVER FAILED IS NOT A GATE
------------------------------------------
This one currently finds nothing, which is exactly the condition
`check_doc_headings.py` and its neighbours exist to warn about. So the proof is
part of the file: `--self-test` feeds it a variable nothing reads and a variable
that is read only through a dynamic prefix, and requires exactly one failure and
one pass. Adding a flag to compose that the code ignores is now caught.

Returns 1 if a variable is set and unreachable, 0 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STACK = ROOT.parent
DEFAULT_COMPOSE = STACK / "docker-compose.yml"

SOURCES = ("main.py", "camera_client.py")

SERVICE = re.compile(r"^  ([a-z0-9_-]+):")
ENV_LINE = re.compile(r"^\s+-\s+([A-Z][A-Z0-9_]*)=")
# `os.getenv("NAME"`, `os.environ.get("NAME"`
LITERAL = re.compile(r'(?:getenv|environ\.get)\(\s*f?"([A-Z][A-Z0-9_]*)"')
# `f"NAME_{` — the per-room scheme. No closing quote required: the next character
# is `{`, which is the whole reason the first version of this regex found nothing.
PREFIX = re.compile(r'f"([A-Z][A-Z0-9_]*_)')


def service_env(compose: str, service: str) -> list[str]:
    """Variable names in one service's `environment:` block, in file order."""
    out: list[str] = []
    inside = False
    for line in compose.splitlines():
        m = SERVICE.match(line)
        if m:
            inside = m.group(1) == service
            continue
        if not inside:
            continue
        e = ENV_LINE.match(line)
        if e:
            out.append(e.group(1))
    return out


def reachable(code: str) -> tuple[set[str], set[str]]:
    literals = set(LITERAL.findall(code))
    prefixes = set(PREFIX.findall(code))
    return literals, prefixes


def unread(env: list[str], code: str) -> list[str]:
    literals, prefixes = reachable(code)
    bad = []
    for name in env:
        if name in literals:
            continue
        if any(name.startswith(p) for p in prefixes):
            continue
        bad.append(name)
    return bad


# --------------------------------------------------------------------------
# Self-test. One that must fail, three that must pass.
# --------------------------------------------------------------------------
_CODE = '''
import os
os.getenv("CAMERA_STREAMS")
os.environ.get("GATEWAY_LOG_DIR", "")
name = "kitchen"
os.getenv(f"CAMERA_PAUSE_ENDPOINT_{name.upper()}")
os.getenv(f"CAMERA_VOSK_DUMP_{name.upper()}")
'''

# The same name was in both lists in the first version, which is a broken
# fixture rather than a broken gate: it demanded that one value both fail and
# pass. The self-test caught it, which is what it is for.
_MUST_FAIL = ["TYPO_FLAG_THAT_DOES_NOTHING", "CAMERA_PAUSE_RAITIO_KITCHEN"]
_MUST_PASS = [
    "CAMERA_STREAMS",                # literal
    "GATEWAY_LOG_DIR",               # os.environ.get, the other literal form
    "CAMERA_PAUSE_ENDPOINT_KITCHEN", # built as f"CAMERA_PAUSE_ENDPOINT_{name...}"
    "CAMERA_VOSK_DUMP_ENTRANCE1",    # same, another room
]


def self_test() -> int:
    failures: list[str] = []
    for name in _MUST_FAIL:
        if name not in unread([name], _CODE):
            failures.append(f"MUST FAIL, but passed: {name}")
    for name in _MUST_PASS:
        if name in unread([name], _CODE):
            failures.append(f"MUST PASS, but flagged: {name}")
    for f in failures:
        print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"self-test: {len(failures)} failure(s) — the gate cannot be trusted",
              file=sys.stderr)
        return 1
    print(f"self-test: {len(_MUST_FAIL) + len(_MUST_PASS)} samples, both "
          "directions, including the dynamic-prefix path")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    compose_path = DEFAULT_COMPOSE
    if "--compose" in argv:
        try:
            compose_path = Path(argv[argv.index("--compose") + 1])
        except IndexError:
            print("  --compose needs a path", file=sys.stderr)
            return 1

    if not compose_path.is_file():
        print(f"  no such file: {compose_path}", file=sys.stderr)
        return 1

    missing_sources = [n for n in SOURCES if not (ROOT / n).is_file()]
    if missing_sources:
        print(f"  missing source(s): {', '.join(missing_sources)}", file=sys.stderr)
        return 1
    code = "".join((ROOT / n).read_text(encoding="utf-8", errors="ignore")
                   for n in SOURCES)

    env = service_env(compose_path.read_text(encoding="utf-8"), "voice_gateway")
    if not env:
        print("  no voice_gateway environment block found — is the service "
              "still named that?", file=sys.stderr)
        return 1

    bad = unread(env, code)
    if bad:
        print(f"  ENV WIRING — {len(bad)} variable(s) set but never read:")
        for name in bad:
            print(f"    {name}")
        print("  A variable nobody reads does nothing, silently.")
        return 1
    print(f"  voice_gateway: all {len(env)} environment variables are reachable "
          "in the code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))