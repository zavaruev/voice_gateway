#!/usr/bin/env python3
"""Gate for one repository rule: prose is English, Cyrillic is data.

Run from the repository root:  python3 scripts/check_lang.py [--self-test]

WHY THIS FILE EXISTS
--------------------
The project's convention has always been "code and comments are in English
(Russian string literals kept for TTS/STT data)", and it was never checked by
anything.  Measured 15.10.2026 when the rule was finally enforced: 217 lines of
Russian *prose* in code across 20 files, and 244 more in AGENTS.md, while
README.md and docs/REFERENCE.md were already clean.

The distinction that makes this check possible at all
----------------------------------------------------
Russian in this repository is not a defect — it is the payload.  This gateway
transcribes Russian speech, speaks Russian, and its field log quotes what the
rooms actually heard («компьютер», «включи свет», «и училась»).  Those are data
and they stay.  What must be English is the *prose around them*.

So the rule is: strip quoted spans, then require the remaining words on a line
to be English.  A naive "does this line contain Cyrillic?" check flags
`# catches the "прощение, спасибо" class of hallucinations`, which is a
perfectly good English sentence, and a check that always fails is a check that
gets switched off — the failure mode this repository has already paid for twice.

A line counts as Russian prose when, after quotes are removed, it has at least
4 Cyrillic letters AND at least as many Cyrillic letters as Latin ones.  Both
halves matter: the count keeps identifiers and stray accents out, and the ratio
is what a bilingual sentence fails — real Russian prose is overwhelmingly
Cyrillic, and English prose quoting Russian is the reverse.

SHELL SCRIPTS ARE THE EXCEPTION, and the self-test found it
----------------------------------------------------------
In Python, markdown and YAML a quoted span is a *citation* of data.  In a shell
script `"…"` is what the program *prints*: `echo "Готово: скрипт отказался
писать"` is output the operator reads, not a quote of a Russian transcript, and
stripping the quotes makes the checker blind to exactly the messages it is meant
to police.  So quotes are stripped everywhere except `*.sh`.  That distinction
is a consequence of the languages, not a convenience, and it is the reason the
self-test is not optional.

SELF-TEST FIRST, ALWAYS
-----------------------
A language gate that has never been seen failing is indistinguishable from a
gate that cannot fail, which is the exact failure mode recorded in
`check_doc_headings.py` next door.  `--self-test` therefore feeds the checker
eight samples: three that MUST be flagged (unquoted Russian prose in a comment,
in a shell echo and in markdown), five traps that must NOT be flagged — quoted
Russian in a comment, Russian inside a fenced code block, a Russian identifier
next to an identifier, a Russian string literal argument, and a stray
abbreviation under the minimum count.

Returns 1 if it found anything, 0 if clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")

MIN_CYRILLIC = 4  # below this it is an identifier, a brand name or an accent

# A triple-quoted block opened anywhere on this line starts a string literal.
TRIPLE = re.compile(r"(?:[frbu]{0,2})(\"\"\"|''')")


def multiline_quote_lines(text: str) -> set[int]:
    """Line numbers that continue a quote opened on an EARLIER line.

    Per-line quote stripping cannot see this: `honesty.py` opens «Хорошо, at the
    end of one comment line and closes it four lines later, so both lines look
    like unquoted Russian prose and both are reported. Flagging them means either
    rewriting a citation or disabling the gate, and the citation is the correct
    answer — it is a field transcript, i.e. data.

    A quote that opens AND closes on one line is not tracked here; the
    per-line strip already removed it before this runs.
    """
    openers = {"«": "»", "“": "”"}
    lines: set[int] = set()
    pending: str | None = None
    for n, raw in enumerate(text.splitlines(), 1):
        if pending is not None:
            lines.add(n)
            if pending in raw:
                pending = None
            continue
        for start, end in openers.items():
            if raw.count(start) > raw.count(end):
                pending = end
                lines.add(n)
                break
    return lines


def _blank_preserve_lines(text: str, keep: set[int]) -> str:
    """Blank every line whose text is inside a string literal, keeping line count.

    A docstring is prose and is in `keep`. An assigned string — the L2 system
    prompt, the tool docstrings the model reads, a stop-word table — is DATA and
    translating it changes model behaviour, so it is blanked before the prose
    test ever sees it. Line numbers must survive, or every reported location is
    off by however many lines the literal was long.
    """
    out: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        out.append(line if n in keep else re.sub(r"[^\n]", " ", line))
    return "\n".join(out)


def python_string_lines(src: str) -> set[int] | None:
    """Lines occupied by a string literal that is NOT prose.

    Three kinds of string, three verdicts:

    * an **assigned** string — the L2 system prompt, a stop-word table — is DATA
      and must not be translated: the model reads it, so translating it changes
      model behaviour;
    * a **docstring** is prose by default, because it is written for whoever
      reads the source next;
    * a docstring on a function decorated with `@tool` is the tool DESCRIPTION
      handed to smolagents — data again, and the reason this function exists.
      `tools.py` even carries a `PROMPT-TEXT WARNING` header saying so.

    Returns None when the file does not parse, so the caller can fall back to
    the line heuristic instead of silently skipping a file it did not understand.
    """
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    def _is_tool(node) -> bool:
        """True when any decorator on `node` is a bare `@tool`."""
        for dec in getattr(node, "decorator_list", []):
            target = dec.func if isinstance(dec, ast.Call) else dec
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name == "tool":
                return True
        return False

    def _docstring_of(node):
        body = getattr(node, "body", None)
        if not body:
            return None
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            return first
        return None

    # Prose docstrings only: skip the ones that are really tool descriptions.
    prose_doc: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_tool(node):
            continue
        doc = _docstring_of(node)
        if doc is not None:
            prose_doc |= set(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))

    str_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            str_lines |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.JoinedStr):
            # An f-string: its literal parts are data, its holes are code.
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    str_lines |= set(range(part.lineno,
                                           (part.end_lineno or part.lineno) + 1))
    return str_lines - prose_doc


SOURCE, SHELL = "source", "shell"

# Citations, in both modes: guillemets, curly quotes, backticks.
CITED = re.compile(r"«[^»]*»|“[^”]*”|`[^`\n]*`")

# Everything else, in source mode only. An ASCII `"…"` in a Python comment or a
# markdown line is a citation of what was heard or said.
ALSO_STRIPPED = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'')


def line_is_russian_prose(line: str, mode: str = SOURCE) -> bool:
    """True when the *prose* of this line is Russian rather than English.

    The two modes differ in what counts as a citation, and the difference is
    forced by the languages:

    * `source` (Python, markdown, YAML) — an ASCII-quoted span is a citation, so
      `catches the "прощение, спасибо" class` is an English sentence and is fine.
    * `shell` — `"…"` is what the program PRINTS, not what it quotes. Stripping
      it is precisely the blindness that let `echo "Готово: скрипт отказался
      писать"` through, so in shell mode only guillemets and backticks are
      stripped and the double quotes stay for the count to see.

    In shell mode the Latin ratio is dropped as well: `echo "=== 1/5 проверка (
    gating stage ) ==="` carries as much Latin as Cyrillic, and it is still a
    Russian message being printed at an English-speaking operator.
    """
    if mode == SHELL:
        bare = CITED.sub(" ", line)
        cyr = len(CYRILLIC.findall(bare))
        return cyr >= MIN_CYRILLIC
    bare = ALSO_STRIPPED.sub(" ", CITED.sub(" ", line))
    cyr = len(CYRILLIC.findall(bare))
    if cyr < MIN_CYRILLIC:
        return False
    return cyr >= len(LATIN.findall(bare))


def scan_text(
    text: str, skip_fences: bool = True, mode: str = SOURCE,
    is_python: bool = False,
) -> list[tuple[int, str]]:
    """Return (line number, line) for every Russian-prose line in `text`.

    `skip_fences` ignores fenced code blocks. They are deliberate: this project's
    docs quote real log output verbatim, and log output in this deployment is
    Russian. Rewriting a quoted log line would falsify the record.

    `is_python` switches on AST-based string-literal removal, which is exact
    where the per-line quote heuristic only guesses.
    """
    if is_python:
        lit = python_string_lines(text)
        if lit is not None:
            text = _blank_preserve_lines(text, keep=_all(text) - lit)
        # else: unparseable — fall through to the line heuristic below.

    # A quote spanning several lines is a citation too, whatever the language.
    continued = multiline_quote_lines(text)

    bad: list[tuple[int, str]] = []
    in_fence = False
    for n, raw in enumerate(text.splitlines(), 1):
        if skip_fences and raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or n in continued:
            continue
        if line_is_russian_prose(raw, mode):
            bad.append((n, raw.strip()))
    return bad


def _all(text: str) -> set[int]:
    return set(range(1, len(text.splitlines()) + 1))


# --------------------------------------------------------------------------
# Self-test. Two directions, and the second is the one people skip.
# --------------------------------------------------------------------------
MUST_FLAG = [
    # A Russian comment with no quoted span at all — the actual defect.
    "# Проверка по РАЗОБРАННОМУ YAML и по КОНКРЕТНОЙ камере.",
    "# 1. имя потока в go2rtc",
    # Markdown prose, unquoted.
    "Полная проверка перед тем, как включить endpointing в бою.",
]
MUST_FLAG_SHELL = [
    # Same text, judged as a shell line: flagged on count, not on ratio.
    'echo "Готово: скрипт отказался писать"',
]
MUST_NOT_FLAG = [
    # English prose quoting Russian. This is the trap: it is a defect neither.
    '# catches the "прощение, спасибо за просмотр" class of hallucinations',
    '# «компьютер, включи свет» -> the camera answered «Да?».',
    # A Russian identifier, and a Russian string literal argument: data.
    'MODEL = wake_features.load("компьютер")',
    'STOPS = ("подожди", "один момент")',
    # Under the minimum count: an abbreviation, not a sentence.
    "# TODO: см. ISSUE-12",
    "",
]


def self_test() -> int:
    failures: list[str] = []

    for sample in MUST_FLAG:
        if not line_is_russian_prose(sample):
            failures.append(f"MUST FLAG, but passed: {sample!r}")

    for sample in MUST_FLAG_SHELL:
        if not line_is_russian_prose(sample, mode=SHELL):
            failures.append(f"MUST FLAG as a shell line, but passed: {sample!r}")

    for sample in MUST_NOT_FLAG:
        if line_is_russian_prose(sample):
            failures.append(f"MUST NOT FLAG, but flagged: {sample!r}")

    # The shell exception must hold through the whole pipeline, not just the
    # line predicate: the same Russian string is data in Python and output in
    # a shell script, and the checker has to tell those apart on its own.
    sh = 'echo "Готово: скрипт отказался писать"\n'
    py = 'TOPS = ("подожди", "один момент")\n'
    if not scan_text(sh, mode=SHELL):
        failures.append("a Russian shell echo was not reported")
    if scan_text(py):
        failures.append("a Russian Python string literal was reported")

    # A shell line that mixes Russian output with English prose in the SAME
    # string is still Russian output. The Latin ratio lets it through, which is
    # how `=== 1/5 проверка ( gating stage ) ===` survived the first version.
    mixed = 'echo "=== 1/5 проверка ( gating stage ) ==="\n'
    if not scan_text(mixed, mode=SHELL):
        failures.append("a mixed Russian shell echo was not reported")

    # ...while a guillemet-wrapped citation inside a shell comment is data.
    cited = '# the room answered «включи свет в коридоре» three times\n'
    if scan_text(cited, mode=SHELL):
        failures.append("a guillemet citation in a shell comment was reported")

    # A fenced block must be invisible even when its contents are Russian:
    # the field log quotes real log output, which here is Russian.
    fenced = "before\n```\nWhisper OK: 'и училась'\nвключи свет\n```\nafter\n"
    if scan_text(fenced):
        failures.append("a fenced code block was scanned for prose")

    # A citation that opens on one line and closes four lines later. The real
    # case is `honesty.py`: «Хорошо, on one line, гостиной» on the next, and
    # both lines look unquoted to a per-line check.
    spanning = ("# the model answered «Хорошо,\n"
                "# выключаю свет в гостиной» in the same step.\n")
    if scan_text(spanning):
        failures.append("a citation spanning lines was reported as prose")

    # The prompt case, which the first version of this gate got wrong: an
    # ASSIGNED multi-line string is data the model reads. Translating the L2
    # system prompt or a tool docstring would change model behaviour, so a
    # checker that demands it is a checker that has to be switched off — and a
    # checker that is switched off catches nothing.
    prompt = (
        'TASK = """Пользователь сказал голосом.\n'
        "Решай задачу по шагам, вызывая тулы.\n"
        'Не выдумывай результат."""\n'
    )
    if scan_text(prompt, is_python=True):
        failures.append("an assigned Russian prompt was reported as prose")

    # The same Russian text as a DOCSTRING is prose, and must be flagged: that
    # is the source of `camera_client.py` and `tools.py`, so the distinction is
    # load-bearing in both directions.
    docstring_src = 'def f():\n    """Вернуть текущее состояние устройства.\n\n    Ничего не выдумывать.\n    """\n    return None\n'
    if not scan_text(docstring_src, is_python=True):
        failures.append("a Russian docstring was NOT reported as prose")

    # ...and the same docstring under `@tool` is the description smolagents
    # hands the model, so it must NOT be flagged. This is the case that made
    # `tools.py` show 44 lines, all of which are tool documentation.
    tool_src = (
        'from smolagents import tool\n\n\n@tool\ndef ha_action(tool_name: str) -> str:\n'
        '    """Выполнить действие в Home Assistant.\n\n'
        '    Вызывай только для действия.\n    """\n    return ""\n'
    )
    if scan_text(tool_src, is_python=True):
        failures.append("an @tool docstring was reported as prose")

    # A file that does not parse must not be skipped silently.
    broken = "def f(:\n    # Проверка по разобранному YAML.\n"
    if not scan_text(broken, is_python=True):
        failures.append("an unparseable Python file was skipped silently")

    # And the whole pipeline: a document that is clean must produce nothing.
    clean = "# Camera\n# Feeds «включи свет» to Whisper.\n"
    if scan_text(clean):
        failures.append("a clean document was reported as dirty")

    for f in failures:
        print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"self-test: {len(failures)} failure(s) — the gate cannot be trusted",
              file=sys.stderr)
        return 1
    print(
        f"self-test: {len(MUST_FLAG) + len(MUST_FLAG_SHELL) + len(MUST_NOT_FLAG) + 9} samples, both "
        "directions, the shell exception, and the prompt/docstring split"
    )
    return 0


def targets() -> list[Path]:
    """Every file the rule applies to.

    `config/` holds ONNX models and `devices.json`; `docs/images/` is SVG. Neither
    is prose. `.venv`, `__pycache__` and build output are not in the repo.
    """
    out: list[Path] = []
    for pat in ("*.py", "*.sh", "*.md", "*.yml", "*.yaml", "*.cfg", "*.toml",
                "tests/*.py", "services/**/*.py", "services/**/*.md",
                "scripts/*.py", "scripts/*.sh", "docs/*.md", ".github/**/*"):
        for p in sorted(ROOT.glob(pat)):
            if p.is_file():
                out.append(p)
    # De-duplicate: a nested match can be reached by two globs.
    seen: set[Path] = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    total = 0
    for path in targets():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {path.relative_to(ROOT)}: unreadable ({exc})", file=sys.stderr)
            continue
        bad = scan_text(
    text,
    mode=SHELL if path.suffix == ".sh" else SOURCE,
    is_python=path.suffix == ".py",
)
        if bad:
            rel = path.relative_to(ROOT)
            total += len(bad)
            print(f"{rel}: {len(bad)} line(s) of Russian prose")
            for n, line in bad:
                print(f"  {n}: {line[:100]}")
    if total:
        print()
        print(f"check_lang: {total} line(s). Prose is English; quoted Russian is data.")
        return 1
    print("check_lang: clean — prose is English everywhere it is prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))