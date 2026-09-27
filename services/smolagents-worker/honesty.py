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

--- ROLE IN THE CASCADE / WHY THIS FILE IS PROGRAMMATIC ---
ESP32 satellite -> main.py (:6050) -> jev-router (L1 :8091) -> smolagents-
worker (L2 :8092, this package) -> Hermes (L3) -> Home Assistant. The L2
CodeAgent runs on a FREE LLM, and the empirical field fact about it is the
single most important thing about this service: the model often writes
`final_answer` in the SAME code block as its tool call, i.e. it announces
success before the tool ever ran. Prompt rules alone were ignored in 3/3
field cases, so honesty is enforced HERE, in code, and the prompt rules in
app.py's TASK_TEMPLATE are only a supporting layer. Do not "simplify" this
module away in favour of better prompting — that was already tried.

CONTRACT
    Inputs : `answer` — the raw draft string returned by agent.run();
             `events`  — the side-effect log recorded by tools.py
                         (`{"tool", "ok", "detail"}` per ha_action call,
                         and `{"tool", "detail"}` per weather_forecast).
    Output : `(answer, replaced)` — the string to actually speak plus a
             flag for logging; `replaced=True` means the veto fired.
    Callers : app.py `_run_agent()` runs vet_answer() first, then
             vet_weather(); nothing else in the pipeline re-checks the text.
    Fail-open : no events / at least one confirmed success / an honest
             admission of failure => the answer passes untouched. The veto
             only fires on PROOF (>=1 recorded failure, zero successes, a
             success claim, no admission). It must never turn a real
             success or an honest refusal into a different sentence.
    Pure stdlib (re only) — unit-testable on the host without
    smolagents/FastAPI (tests/test_honesty.py); the regexes below are
    behaviour-critical, their pattern strings are matched against real
    field transcripts and must not be "tidied".
"""

from __future__ import annotations

import re

# Success claims about an action (what must not be said without a result).
# Field regression 25.09.2026: «Хорошо, отправляю робота-пылесоса на кухню!»
# was spoken after vacuum__HassVacuumStart failed with MatchFailedReason.AREA
# — «отправляю/запускаю/начинаю/убираю» were missing from this list. The
# (?<!не ) guard keeps honest negations («робот не убирается») out of the
# claim set: a negated verb is a status report, not a promise.
_RE_CLAIM = re.compile(
    r"(?<!не )\b(?:включ\w*|выключ\w*|зажг\w*|погас\w*|сдела(?:л\w*|но|на)|"
    r"выполнено|запущен\w*|запусти\w*|запуска\w*|подготовлен\w*|передала|"
    r"активир\w*|открыл\w*|закрыл\w*|отправ\w*|начина\w*|убира\w*|убер[её]т\w*)\b",
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
    """Plain-Russian refusal built from the last recorded failure.

    Walks the event log newest-first and picks the most recent failed call;
    the returned sentence maps the HA error marker in `detail` to a natural
    Russian phrase (TTS-ready, 1 short sentence). Unknown markers fall
    through to the generic refusal — never invent a success here.
    """
    detail = ""
    for e in reversed(events):
        if not e.get("ok"):
            detail = str(e.get("detail", ""))
            break
    if "ASSISTANT" in detail:
        # Entity exists but is not exposed to the voice assistant: no
        # wording about rooms/names would help, tell the real blocker.
        return (
            "Не получилось: устройство не открыто голосовому ассистенту — "
            "им нельзя управлять голосом."
        )
    if "MatchFailedReason.NAME" in detail or "No exposed" in detail:
        return "Не нашла такого устройства в Home Assistant."
    if "INVALID_AREA" in detail:
        return "Не нашла такую комнату в Home Assistant."
    if "MatchFailedReason.AREA" in detail:
        # The room exists, but no device in it matched — for the vacuum this
        # means it is not assigned to any area in HA (HassVacuumStart's area
        # slot filters by the vacuum's LOCATION, not the cleaning target).
        return ("Не получилось: устройство не привязано к этой комнате "
                "в Home Assistant.")
    if "Некорректный arguments_json" in detail:
        return "Не получилось: команда собрана с ошибкой, попробуй переформулировать."
    return "Не получилось: Home Assistant отклонил команду."


def vet_answer(answer: str, events: list[dict]) -> tuple[str, bool]:
    """-> (answer, replaced). Replaces a success claim made after only
    failed ha_action calls with the recorded truth.

    Gates, in order (ALL must pass for the veto to fire):
      1. >=1 recorded ha_action event (otherwise nothing to prove against);
      2. zero events with ok=True (one confirmed side effect = trust it);
      3. _RE_CLAIM finds a success claim in the answer;
      4. _RE_FAIL finds NO admission of failure (honest refusals pass).
    On fire the whole answer is swapped for `_truth(events)` — a short,
    factual, speakable sentence; returns replaced=True for logging.
    """
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
    """Detail of the LAST recorded weather_forecast call (the run may call
    it more than once; the newest output is the current truth)."""
    return str(events[-1].get("detail", "")) if events else ""


def _forecast_faithful(answer: str, forecast: str) -> bool:
    """Does the answer reproduce the recorded forecast (numbers + words)?"""
    # Normalise case and ё->е so «обла́чность»/«Облачность» compare equal.
    ans = answer.lower().replace("ё", "е")
    fc = forecast.lower().replace("ё", "е")
    # Every distinct number in the forecast must appear in the answer.
    # dict.fromkeys() de-duplicates while keeping order; the (?<!\d)…(?!\d)
    # lookarounds stop a partial hit ("16" inside "160") counting as a match.
    for n in dict.fromkeys(re.findall(r"\d+", fc)):
        if not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", ans):
            return False
    # Weather-condition stems present in the forecast must survive into the
    # answer («пасмурно» must not become «солнечно»); absence in the
    # forecast is fine — the tool may simply not have mentioned rain.
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
    # Skip when a ha_action was attempted: the replacement below is
    # whole-answer and must never wipe an action report (see header).
    if not events or action_attempted:
        return answer, False
    forecast = _recorded_forecast(events)
    if not forecast:
        return answer, False
    # Did the tool itself fail? (its error text is recorded verbatim)
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
        return answer, False  # numbers + condition words carried over: honest
    # Distorted or invented forecast — speak the recorded tool output
    # verbatim (less conversational, but always true).
    return forecast, True
