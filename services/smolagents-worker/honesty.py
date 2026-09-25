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

Weather veto (field regression 25.09.2026): the free model called
weather_forecast AND final_answer in the SAME code block — i.e. it wrote
the forecast before reading the tool output and then distorted it anyway
(tool: «пасмурно до 16°/9°», answer: «переменная облачность +22.5/+12.1»).
Prompt prohibitions were ignored twice out of two, so the recorded tool
output decides: a recorded forecast missing from the answer replaces it,
and a forecast claim after a failed fetch is replaced with the refusal.
Skipped when a ha_action was attempted in the same run — the replacement
is whole-answer and must not wipe an action report (that combination
needs a garbled compound utterance, effectively never).
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


# --- Weather veto -----------------------------------------------------------
# The free model writes final_answer BEFORE reading the weather_forecast
# output (same code block), so the recorded tool detail — not the answer —
# is the source of truth. False positives are harmless: the replacement is
# the verbatim recorded forecast, always true, only less conversational.

# Condition stems that must carry over from the tool output to the answer.
_WCOND = (
    "солнеч", "ясн", "пасмур", "облач", "дожд", "ливн", "снег", "гроз",
    "туман", "морос", "град",
)

# The answer talks about weather at all (used for the failure branch).
_RE_W_CLAIM = re.compile(
    r"\d+\s*(?:°|градус\w*)|градус\w*|температур\w*|погод\w*|"
    r"солнечно|пасмурно|облачн\w*|дожд\w*|снег\w*|гроз\w*|ветер|осадк\w*",
    re.IGNORECASE,
)

# The answer already admits the fetch failed — nothing to veto.
_RE_W_FAIL = re.compile(
    r"недоступн\w*|не\s+смог\w*|не\s+удалось|нет\s+данных|не\s+получилось|"
    r"без\s+прогноза|не\s+ответил",
    re.IGNORECASE,
)


def _recorded_forecast(events: list[dict]) -> str:
    return str(events[-1].get("detail", "")) if events else ""


def _forecast_faithful(answer: str, forecast: str) -> bool:
    """Does the answer reproduce the recorded forecast (numbers + words)?"""
    ans = answer.lower().replace("ё", "е")
    fc = forecast.lower().replace("ё", "е")
    for n in dict.fromkeys(re.findall(r"\d+", fc)):
        if not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", ans):
            return False
    for stem in _WCOND:
        if stem in fc and stem not in ans:
            return False
    return True


def vet_weather(
    answer: str, events: list[dict], action_attempted: bool = False
) -> tuple[str, bool]:
    """-> (answer, replaced). Enforces the recorded weather_forecast output.

    Skipped entirely when a ha_action was attempted in the same run — the
    replacement is whole-answer, so it must never wipe an action report
    (true success or honest refusal). That combination needs a garbled
    compound utterance, effectively never.
    """
    if not events or action_attempted:
        return answer, False
    forecast = _recorded_forecast(events)
    if not forecast:
        return answer, False
    failed = bool(
        re.search(r"недоступен|не\s+удалось|ошибк", forecast, re.IGNORECASE)
    )
    if failed:
        # No forecast on record: any weather claim is fabricated, but a
        # plain restatement of "couldn't get it" must pass through.
        if _RE_W_CLAIM.search(answer) and not _RE_W_FAIL.search(answer):
            return forecast, True
        return answer, False
    if _forecast_faithful(answer, forecast):
        return answer, False
    return forecast, True
