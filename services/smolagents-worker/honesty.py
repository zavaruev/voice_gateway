"""Deterministic honesty veto for the L2 final answer (smolagents-worker).

Field regressions (E2E, Sep 2026): the free model answered «Хорошо,
выключаю свет в гостиной» / «Кафеварка включена!» in the SAME step where its
own ha_action returned MatchFailedError — prompt prohibitions were ignored
3 times out of 3. A promise without a performed side effect violates the
project rule, so the spoken answer is checked against the *recorded* tool
outcomes and a success claim is replaced with the recorded truth.

Veto fires only on proof: >=1 recorded failure of the claim's own kind
(ha_action for a promise, ha_read for a state/number), zero successes of
that kind, the claim in the answer, and no admission of failure. Pure
stdlib — unit-testable on the host without smolagents/FastAPI.

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
                         and `{"tool", "detail"}` per weather_forecast);
             `read_events` — the same shape for ha_read calls (a READ has
                         no side effect, so it lives in its own buffer and
                         proves DATA claims only, never actions);
             `weather_events` — optional, for GROUNDING only: a degree or
                         percentage may legitimately come from the forecast
                         instead of a sensor read.
    Output : `(answer, replaced)` — the string to actually speak plus a
             flag for logging; `replaced=True` means the veto fired.
             `failure_note(events + read_events)` is the sibling helper:
             the RAW recorded errors for app._run_agent's one bounded retry
             after a veto (the speakable `_truth()` above would teach the
             model nothing).
    Siblings: `same_block_check()` is the structural guard smolagents calls
             through `final_answer_checks` BEFORE accepting an answer: it
             refuses an answer written in the same code block as a tool
             call, so the text can only come from the tool output that is
             already in the model's context. `mark_tool_call()` /
             `reset_tool_call()` / `reset_run_state()` keep its flag in step
             with tools.py and app.py.
    Callers : app.py `_run_agent()` runs vet_answer() first, then
             vet_weather(); nothing else in the pipeline re-checks the text.
    Fail-open : no events / a confirmed result of the claim's own kind / an
             honest admission of failure => the answer passes untouched.
             The veto only fires on PROOF (>=1 recorded failure of that
             kind, zero successes, a claim, no admission). It must never
             turn a real success or an honest refusal into a different
             sentence.
    Pure stdlib (re only) — unit-testable on the host without
    smolagents/FastAPI (tests/test_honesty.py); the regexes below are
    behaviour-critical, their pattern strings are matched against real
    field transcripts and must not be "tidied".
"""

from __future__ import annotations

import re

# --- Same-block guard (structural, via smolagents `final_answer_checks`) -----
# The empirical fact this whole module exists: the free model writes
# `final_answer` in the SAME code block as its tool call, i.e. it announces
# its answer while the tool output does not exist yet — the text can then
# only come from imagination or from the dialogue history. Field cases:
# 25.09 (forecast written before the fetch), 29.09 11:39 («заряд 45 %» after
# a failed read) and 29.09 14:17 («робот убирает в гостиной, заряд батареи
# шестьдесят пять процентов» — the read was on screen and carried NEITHER
# state nor battery). The prompt forbids it in words and was ignored every
# single time, so it is refused here: smolagents runs `final_answer_checks`
# before accepting an answer, and raising inside one turns into an
# AgentError that is recorded ON the step while the run CONTINUES — next
# step, the tool output is in the model's context, which is where the answer
# must come from.
#
# `mark_tool_call()` is called by every tool in tools.py; `reset_tool_call()`
# by app.py's step callback at the end of each step; `reset_run_state()` by
# app._run_agent at the start of every attempt (next to the event buffers).

_TOOL_IN_STEP = False  # a real tool ran inside the code block being executed
_BLOCK_GUARD_USED = False  # fire once per run (see module logic above)


def mark_tool_call() -> None:
    """A tool started executing inside the current code block."""
    global _TOOL_IN_STEP
    _TOOL_IN_STEP = True


def reset_tool_call() -> None:
    """End of step: the next code block starts with a clean flag."""
    global _TOOL_IN_STEP
    _TOOL_IN_STEP = False


def reset_run_state() -> None:
    """Per attempt: no tool called yet, guard re-armed."""
    global _TOOL_IN_STEP, _BLOCK_GUARD_USED
    _TOOL_IN_STEP = False
    _BLOCK_GUARD_USED = False


def same_block_check(final_answer, memory, agent=None) -> bool:
    """`final_answer_checks` hook for CodeAgent — refuse an answer written in
    the same code block as a tool call.

    Returns True when the answer may be spoken. Otherwise it raises, and
    smolagents wraps that into an AgentError: the step is recorded as an
    error and the loop goes on, so the run does not die. The message IS what
    the model reads on the next step, hence an instruction, not a log line.
    Signature is smolagents' contract: `check(final_answer, memory, agent)`.

    Fires once per run (`_BLOCK_GUARD_USED`): a model that repeats itself
    would otherwise spend MAX_STEPS on refusals — the vetoes below still
    cover the second attempt, so nothing is lost by letting it through.
    """
    global _BLOCK_GUARD_USED
    if not _TOOL_IN_STEP or _BLOCK_GUARD_USED:
        return True
    _BLOCK_GUARD_USED = True
    raise AssertionError(
        "Ответ написан в том же блоке кода, что и вызов тула, — то есть до "
        "чтения его вывода. Вывод тула уже есть в журнале шага выше: "
        "прочитай его и вызови final_answer отдельно, на следующем шаге, "
        "строго по этим данным, без предположений."
    )

# Success claims about an action (what must not be said without a result).
# Field regression 25.09.2026: «Хорошо, отправляю робота-пылесоса на кухню!»
# was spoken after vacuum__HassVacuumStart failed with MatchFailedReason.AREA
# — «отправляю/запускаю/начинаю/убираю» were missing from this list. The
# (?<!не ) guard keeps honest negations («робот не убирается») out of the
# claim set: a negated verb is a status report, not a promise.
_RE_CLAIM = re.compile(
    r"(?<!не )\b(?:включ\w*|выключ\w*|зажг\w*|погас\w*|сдела(?:л\w*|но|на)|"
    r"выполнено|запущен\w*|запусти\w*|запуска\w*|подготовлен\w*|передала|"
    r"активир\w*|открыл\w*|закрыл\w*|отправ\w*|начина\w*|убира\w*|убер[её]т\w*)\b"
    # Media transport promises (03.10.2026): «поставила на паузу»,
    # «переключила трек», «заглушила» are exactly as much a promise as
    # «включила» and must be vetoed the same way — and «поставила» alone is
    # the model's usual wording. The extra lookbehind keeps the STATUS report
    # «не на паузе» out — the plain `(?<!не )` cannot see two words back.
    r"|(?<!не на )\bпауз\w*|\bпереключ\w*|\bзаглуш\w*|\bприостанов\w*"
    r"|(?<!не )\bпостав\w*",
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

# State/DATA claims: device states and readings the model can only know
# from a ha_read. Field regression 29.09.2026: «Что там с нашим пылесосом?»
# came back as «Пылесос сейчас работает, уровень заряда сорок пять
# процентов» while ha_read returned
# `{"success": false, "error": "No exposed entities matched name 'пылесос'"}`
# — HA reports `docked` and has no battery attribute for that vacuum at all,
# so both halves were invented. The claim is spelled out as WORDS because
# this model writes numbers in words («сорок пять», not «45»); the digit
# branch only covers readings with an explicit unit. Same (?<!не ) guard as
# _RE_CLAIM: «не работает» is a status report, not a claim.
_RE_DATA_CLAIM = re.compile(
    r"(?<!не )\b(?:работает|работал\w*|убирается|убирал\w*|заряжен\w*|заряд\w*|"
    r"батаре\w*|процент\w*|готов\w*|включено|выключено|приставлен\w*)\b"
    r"|\bна\s+базе\b|\bв\s+доке\b"
    r"|\b(?:температур\w*|влажност\w*|громкост\w*|яркост\w*)"
    r"|\b\d{1,3}\s*(?:%|°|градус\w*)",
    re.IGNORECASE,
)


def _truth(events: list[dict]) -> str:
    """Plain-Russian refusal built from the last recorded failure.

    Walks the event log newest-first and picks the most recent failed call;
    the returned sentence maps the HA error marker in `detail` to a natural
    Russian phrase (TTS-ready, 1 short sentence). Unknown markers fall
    through to the generic refusal — never invent a success here.
    """
    fail: dict = {}
    for e in reversed(events):
        if not e.get("ok"):
            fail = e
            break
    detail = str(fail.get("detail", ""))
    if str(fail.get("tool", "")) == "ha_read":
        # A READ produced nothing: there is no state to report, and the
        # action-shaped refusals below («команда отклонена») would describe a
        # command the user never gave.
        if detail.startswith("Не удалось"):
            return "Не получилось: Home Assistant не отвечает на запросы."
        return "Не нашла такого устройства в Home Assistant."
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
    # Media (03.10.2026): an invented MCP tool name and an unresolvable
    # player are different blockers and deserve different sentences — the
    # first one is «я назвала несуществующий тул», which used to be spoken to
    # the user as a vague «Home Assistant отклонил команду».
    if "нет тула" in detail:
        return ("Не получилось: такого действия Home Assistant не умеет — "
                "плеером надо управлять через media_control.")
    if "не нашла медиаплеер" in detail:
        return ("Не нашла медиаплеер, к которому это относится: скажи "
                "комнату или как называется устройство.")
    # A media command the box could not act on because it is not doing
    # anything: «ничего не играет» is the truth, «отклонил команду» is a
    # different (wrong) story — and the veto must not spend the retry pass on
    # it either, which it does not: failure_note feeds a fix, not a re-ask.
    if "ничего не играет" in detail:
        return "Ничего не играет."
    if "уже на паузе" in detail:
        return "Уже на паузе."
    if "уже не играет" in detail:
        return "Уже не играет."
    if "вместо" in detail and "открыл" in detail:
        # Kodi started something OTHER than what was asked for (a plugin
        # redirect, a stale queue). The refusal must name both titles or the
        # user cannot tell what is now on the screen (03.10.2026).
        m = re.search(r"открыл «(.+?)» вместо «(.+?)»", detail)
        if m:
            return f"Включилось «{m.group(1)}» вместо «{m.group(2)}»."
    if "громкость не изменилась" in detail:
        return "Громкость не изменилась."
    if "подходит несколько медиаплееров" in detail:
        return "Подходит несколько медиаплееров — назови комнату."
    return "Не получилось: Home Assistant отклонил команду."


# --- Data GROUNDING (field case 29.09.2026, 14:17) ---------------------------
# The gates in vet_answer() only prove a claim wrong when the read FAILED.
# This turn had a SUCCESSFUL read — ha_read("пылесос") returned ten entity
# lines — and the model still dictated «Робот сейчас убирает в гостиной,
# заряд батареи шестьдесят пять процентов»: the payload carried neither the
# cleaning state nor a battery entity (HA exposes none for that vacuum), and
# the vacuum's own `docked` line had been cut off by match_states' cap.
# A successful read of SOME entities proves nothing about ANY entity, so a
# data claim has to be GROUNDED: its subject must occur in a recorded
# payload, otherwise the claim is as good as invented.
#
# Each rule is (claim pattern, evidence the payload must carry, RU sentence
# to speak instead). Evidence is matched LENIENTLY (latin stems by prefix
# with a non-alnum lookbehind — «charge» must also match «charging», but
# «on» must not be satisfied by «person»; Russian stems as plain
# substrings): a missed match here would replace a TRUE answer, which is
# worse than letting a doubtful one through.
_GROUND: tuple[tuple[re.Pattern[str], tuple[str, ...], str], ...] = (
    (re.compile(r"заряд|заряжен|батаре|процент", re.I),
     ("battery", "заряд", "charge", "%"),
     "Данных о заряде батареи в Home Assistant нет."),
    (re.compile(r"на\s+базе|в\s+доке|пристан\w*", re.I),
     ("docked", "док", "на базе", "в доке"),
     "В Home Assistant нет сведений, что робот на базе."),
    (re.compile(r"убирает|убирается|уборк\w*|чистит|моет", re.I),
     ("cleaning", "возвращается", "в работе", "запущен"),
     "В Home Assistant нет сведений, что робот сейчас убирается."),
    (re.compile(r"температур\w*|градус", re.I),
     ("temperature", "temp", "температур", "°", "градус"),
     "В Home Assistant нет данных о температуре."),
    (re.compile(r"влажност", re.I), ("humidity", "влажн"),
     "В Home Assistant нет данных о влажности."),
    (re.compile(r"громкост", re.I), ("volume", "громкост"),
     "В Home Assistant нет данных о громкости."),
    (re.compile(r"яркост", re.I), ("brightness", "яркост"),
     "В Home Assistant нет данных о яркости."),
    (re.compile(r"включено|горит|работает", re.I),
     ("on", "включ", "playing", "active", "cleaning", "open", "открыт"),
     "В Home Assistant не видно, чтобы это было включено."),
    (re.compile(r"выключено|погашен", re.I),
     ("off", "выключ", "standby"),
     "В Home Assistant не видно, чтобы это было выключено."),
)

_NUMBER_TRUTH = "В Home Assistant нет таких данных."


def _has_evidence(key: str, pool: str) -> bool:
    """Is `key` present in the (already lowercased) payload pool?"""
    if not key.isascii():
        return key in pool
    return re.search(rf"(?<![a-z0-9_]){re.escape(key)}", pool) is not None


def _payload_pool(
    reads: list[dict], weather_events: list[dict] | None
) -> str:
    """Lowercased concatenation of every payload this run may quote: the
    successful ha_read details plus the recorded weather_forecast output
    («15 градусов» may legitimately come from the forecast, not from a
    sensor read, and must never be vetoed as ungrounded)."""
    parts = [str(e.get("detail", "")) for e in reads if e.get("ok")]
    parts += [str(e.get("detail", "")) for e in (weather_events or [])]
    return "\n".join(parts).lower().replace("ё", "е")


def _grounding_truth(answer: str, pool: str) -> str | None:
    """-> the RU sentence to speak instead of `answer`, or None when every
    data claim in it is carried by a recorded payload.

    When several claims fail, the one appearing EARLIEST in the answer picks
    the sentence — the user hears about the first thing that was made up.
    Digit percentages are checked against the payload numbers too (a 65 %
    claim cannot stand on a payload that says 82); word numbers («шестьдесят
    пять») are covered by the subject rules instead, since the model spells
    them out.
    """
    if not _RE_DATA_CLAIM.search(answer):
        return None
    failures: list[tuple[int, str]] = []
    for pattern, evidence, sentence in _GROUND:
        m = pattern.search(answer)
        if m and not any(_has_evidence(k, pool) for k in evidence):
            failures.append((m.start(), sentence))
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:%|процент)", answer, re.I):
        val = float(m.group(1).replace(",", "."))
        if not any(abs(val - n) <= 0.5 for n in _payload_numbers(pool)):
            failures.append((m.start(), _NUMBER_TRUTH))
            break
    if not failures:
        return None
    return min(failures, key=lambda f: f[0])[1]


def _payload_numbers(pool: str) -> list[float]:
    """All numbers readable in the payload pool (comma and dot spellings)."""
    out: list[float] = []
    for tok in re.findall(r"\d+(?:[.,]\d+)?", pool):
        try:
            out.append(float(tok.replace(",", ".")))
        except ValueError:  # pragma: no cover — \d+ always parses
            continue
    return out


def vet_answer(
    answer: str,
    events: list[dict],
    read_events: list[dict] | None = None,
    weather_events: list[dict] | None = None,
) -> tuple[str, bool]:
    """-> (answer, replaced). Replaces an unsupported claim with the truth.

    THREE PROOFS, each with its own log (the buffers are kept apart in
    tools.py on purpose — a read must never pass for a side effect):
      * action claim («включил/отправляю/сделал», _RE_CLAIM) -> ha_action log;
      * data claim after a FAILED read («заряд сорок пять процентов»,
        «сейчас работает», _RE_DATA_CLAIM) -> ha_read log. Added 29.09.2026
        for the vacuum turn: the read failed with `success: false` and the
        model still dictated a state and a battery percentage — vet_answer
        only looked at ha_action events, saw an empty log and passed it;
      * data claim after a SUCCESSFUL read -> GROUNDING (see _GROUND): the
        claim's subject must occur in a recorded payload. Added 29.09.2026
        14:17, the second occurrence: ten entities read, «робот убирает,
        заряд 65 %» spoken — neither was in those entities.
    Gates for the first two proofs, in order (ALL must pass to fire):
      1. >=1 recorded event of the relevant kind (nothing to prove against
         otherwise -> fail open);
      2. zero events with ok=True of that kind (one confirmed result =
         trust the model's report);
      3. the claim is present;
      4. no admission of failure (honest refusals pass).
    Grounding needs its own gates: a successful read, NO confirmed action
    (a confirmed side effect backs a state claim — «включил» -> «включено»
    is true without a fresh read), and the claim present.
    On fire the whole answer is swapped for a short, factual, speakable
    sentence (`_truth()` for failures, the rule's sentence for ungrounded
    claims); returns replaced=True for logging.
    """
    reads = list(read_events or [])
    if not events and not reads:
        return answer, False  # nothing recorded: fail open
    if _RE_FAIL.search(answer):
        return answer, False  # already honest about the failure
    acts_ok = any(e.get("ok") for e in events)
    reads_ok = any(e.get("ok") for e in reads)
    if events and not acts_ok and _RE_CLAIM.search(answer):
        # Reads first: _truth() walks newest-first, and an action error
        # (NAME/AREA/ASSISTANT) is the more specific of the two failures.
        return _truth(reads + list(events)), True
    if reads and not reads_ok and _RE_DATA_CLAIM.search(answer):
        return _truth(reads + list(events)), True
    if reads_ok and not acts_ok:
        truth = _grounding_truth(
            answer, _payload_pool(reads, weather_events)
        )
        if truth:
            return truth, True
    return answer, False


def failure_note(events: list[dict]) -> str:
    """-> RU context block for the bounded retry after a veto, or "".

    vet_answer swaps the claim for the user-facing `_truth()` sentence; the
    retry needs the RAW recorded errors instead, because the model has to
    correct its target (device name / room), not just phrase a refusal
    better. Capped at 400 chars: a free model loses the thread in walls of
    MCP JSON, and one failed call is all the signal there is.

    Returns "" when nothing failed — then there is nothing to retry from and
    the caller must not spend a second run (a blind repeat would only add
    latency and another chance to act on a wrong target).
    """
    fails = [str(e.get("detail", "")) for e in events if not e.get("ok")]
    if not fails:
        return ""
    return (
        "Предыдущая попытка этого же запроса провалилась: "
        + "; ".join(fails)[:400]
        + ". Попробуй иначе — другое имя устройства или комнаты — либо прямо "
        "скажи, что не вышло; не повторяй отклонённый вызов один в один и "
        "не заявляй успех без подтверждённого результата тула."
    )


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
