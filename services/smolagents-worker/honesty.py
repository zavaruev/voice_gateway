"""Deterministic honesty veto for the L2 final answer (smolagents-worker).

Field regressions (E2E, Sep 2026): the free model answered «Хорошо,
выключаю свет в гостиной» / «Кафеварка включена!» in the SAME step where its
own ha_action returned MatchFailedError — prompt prohibitions were ignored
3 times out of 3. A promise without a performed side effect violates the
project rule, so the spoken answer is checked against the *recorded* tool
outcomes and a success claim is replaced with the recorded truth.

Veto fires only on proof: >=1 recorded ha_action failure, zero successes, a
success claim in the answer, and no admission of failure. Pure stdlib —
unit-testable on the host without smolagents/FastAPI.
"""

from __future__ import annotations

import re

# Success claims about an action (what must not be said without a result).
_RE_CLAIM = re.compile(
    r"\b(?:включ\w*|выключ\w*|зажг\w*|погас\w*|сдела(?:л\w*|но|на)|выполнено|"
    r"запущен\w*|подготовлен\w*|передала|активир\w*|открыл\w*|закрыл\w*)\b",
    re.IGNORECASE,
)

# The answer already admits failure — truth is told, nothing to veto.
_RE_FAIL = re.compile(
    r"не\s+(?:получилось|удалось|могу|смог\w*|нашл\w*|удал\w*|возможно|выходит|"
    r"получится|получается|выполн\w*)"
    r"|нет\s+такого|не\s+найден|отклонил|отклонена|ошибк\w*|"
    r"недоступн\w*|невозможн\w*|нельзя",
    re.IGNORECASE,
)


def _truth(events: list[dict]) -> str:
    """Plain-Russian refusal built from the last recorded failure."""
    detail = ""
    for e in reversed(events):
        if not e.get("ok"):
            detail = str(e.get("detail", ""))
            break
    if "ASSISTANT" in detail:
        return (
            "Не получилось: устройство не открыто голосовому ассистенту — "
            "им нельзя управлять голосом."
        )
    if "MatchFailedReason.NAME" in detail or "No exposed" in detail:
        return "Не нашла такого устройства в Home Assistant."
    if "INVALID_AREA" in detail:
        return "Не нашла такую комнату в Home Assistant."
    if "Некорректный arguments_json" in detail:
        return "Не получилось: команда собрана с ошибкой, попробуй переформулировать."
    return "Не получилось: Home Assistant отклонил команду."


def vet_answer(answer: str, events: list[dict]) -> tuple[str, bool]:
    """-> (answer, replaced). Replaces a success claim made after only
    failed ha_action calls with the recorded truth."""
    if not events:
        return answer, False  # nothing recorded (no action attempted): fail open
    if any(e.get("ok") for e in events):
        return answer, False  # at least one confirmed side effect
    if not _RE_CLAIM.search(answer):
        return answer, False  # no success claim to veto
    if _RE_FAIL.search(answer):
        return answer, False  # already honest about the failure
    return _truth(events), True
