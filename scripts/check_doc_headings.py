#!/usr/bin/env python3
"""Проверка заголовков AGENTS.md на склейку. ДЕМОНСТРАЦИЯ, ЧТО ОНА ЛОВИТ.

Запускать из корня репозитория:  python3 scripts/check_doc_headings.py

История, почему файл существует. Сплей в AGENTS.md
(`s[:i] + new + s[i:]`, где `new` уже заканчивается якорем) склеивает
заголовок с копией себя же. Пять раз за 14–15.10.2026, и **дважды проверка,
которую я запускал после этого, проходила при наличии дефекта:**

  * первая искала `**- **` — а склейка выглядит как `ФРАЗУ- **АТТЕНЦИОННЫЙ`,
    дефис ПОСЛЕ звёзд, и под шаблон не подходила никогда;
  * вторая ловила `- **` не в начале строки — но склеенная строка как раз
    начинается с `- **`, просто содержит его дважды. Она ловила другой случай
    (текст, приклеенный к концу предыдущей строки), а не этот.

Отсюда правило, которое этот файл существует, чтобы применять: **проверка,
которая ни разу не была замечена падающей на настоящем дефекте, ничего не
значит.** Поэтому `--self-test` ниже сначала ломает строку так же, как ломает
её сплей, и убеждается, что проверка это замечает. Если не замечает — скрипт
падает сам, а не файл.

Три независимых признака, потому что они ловят разное:

  1. `- **` не в начале строки — текст приклеен к концу предыдущей строки;
  2. `- **` встречается в строке больше одного раза — заголовок склеен с копией
     себя же (ровно то, что делает сплей);
  3. два заголовка с одинаковым началом — один и тот же заголовок продублирован
     целиком, в разных строках.

Возвращает 1, если что-то нашёл, иначе 0.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "AGENTS.md"
MARK = "- **"


def _strip_code(line: str) -> str:
    """Убрать ограждённые блоки и инлайновые кавычки.

    Без этого проверка ругается сама на себя: заметка о шаблоне склейки
    цитирует `- **` буквально, и каждая такая цитата выглядит как дефект.
    Проверка, которая всегда падает, тоже бесполезна — её выключают.
    """
    return re.sub(r"`[^`]*`", "", line)


def find_glue(lines: list[str]) -> list[tuple[int, str]]:
    """Три независимых признака склейки, каждый со своим сообщением."""
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
            bad.append((i, f"текст приклеен к концу строки ({n}× «{MARK}» не в начале)"))
        elif n > 1:
            bad.append((i, f"заголовок склеен с копией себя же ({n}× «{MARK}» в строке)"))
    heads = [
        (i, _strip_code(line).lstrip()[len(MARK):].split("**")[0])
        for i, line in enumerate(lines, 1)
        if _strip_code(line).lstrip().startswith(MARK)
    ]
    seen: dict[str, int] = {}
    for num, key in heads:
        key = key[:45]
        if key in seen:
            bad.append((num, f"заголовок повторён (строка {seen[key]}): {key[:45]}"))
        else:
            seen[key] = num
    return bad


def self_test() -> int:
    """Проверяем ПРОВЕРКУ на заведомо битых образцах, до того как ей поверим."""
    cases = [
        ("заголовок склеен с копией себя", "- **A** текст- **A** текст", True),
        ("дефис внутри, не в начале",       "  слово- **A** текст",      True),
        ("один заголовок, чисто",            "- **A** текст",            False),
        ("обычный текст",                    "  просто строка",          False),
        ("два разных заголовка",             "- **A**\n- **B**",         False),
        ("цитата шаблона в строке",          "  признак: `- **` тут",   False),
        ("цитата в ограждённом блоке",       "  ```\n  - **x**- **x**\n  ```", False),
    ]
    fails = 0
    for name, sample, want in cases:
        got = bool(find_glue(sample.splitlines()))
        ok = "ок" if got == want else "НЕ ЛОВИТ"
        if got != want:
            fails += 1
        print(f"  self-test: {name:<38} ждём {str(want):<5} {ok}")
    print(f"  self-test: {'все образцы пройдены' if not fails else f'ПРОВАЛЕНО: {fails}'}")
    return fails


def main() -> int:
    if "--self-test" in sys.argv:
        return 1 if self_test() else 0

    if not DOC.exists():
        print(f"  нет файла: {DOC}")
        return 1
    lines = DOC.read_text(encoding="utf-8").splitlines()
    bad = find_glue(lines)
    heads = sum(1 for l in lines if l.lstrip().startswith(MARK))
    if bad:
        print(f"  СКЛЕЙКА — {len(bad)} шт. в {DOC.name}:")
        for num, why in bad:
            print(f"    строка {num}: {why}")
            print(f"      {lines[num - 1][:120]}")
        return 1
    print(f"  {DOC.name}: заголовков {heads}, склеек нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
