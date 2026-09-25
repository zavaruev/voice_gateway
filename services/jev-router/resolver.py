"""Intent/slot resolver for easy routes.

Maps a Russian voice command to an HA MCP tool call. Returns None whenever the
command is ambiguous or has no MCP counterpart — the caller then escalates to
COMPLEX (L2 CodeAgent), never a wrong side-effect.
"""

import logging
import re
from dataclasses import dataclass, field

import config

logger = logging.getLogger("router.resolver")

# --- Room dictionary: RU keyword -> canonical area name. ---
# names[0] is what goes to the HA intent matcher AND to find_entity/_area_phrase.
# Verified live (Sep 2026) against the area registry: the matcher accepts the
# display names below, but RU aliases exist only for some areas — «кухня»
# resolves, «гостиная» -> MatchFailedError INVALID_AREA (E2E regression).
ROOMS: dict[str, list[str]] = {
    "кухн": ["Kitchen", "кухня"],
    "гостин": ["Living Room", "гостиная"],
    "зал": ["Living Room", "гостиная"],
    "спальн": ["Bedroom", "спальня"],
    "коридор": ["Corridor", "коридор"],
    "прихож": ["Entrance", "прихожая"],
    "ванн": ["Bathroom", "ванная"],
    "туалет": ["WC", "туалет"],
    "детск": ["детская", "kids", "children"],
    "кабинет": ["кабинет", "office"],
    "балкон": ["Balcony", "балкон"],
    "улиц": ["улица", "street"],
    "двор": ["двор", "yard"],
    "подвал": ["подвал", "basement"],
}

# Default area when the speaker does not name one (from the device stream name).
STREAM_DEFAULT_AREA: dict[str, str] = {
    "kitchen": "Kitchen",
    "livingroom": "Living Room",
    "living_room": "Living Room",
    "corridor": "Corridor",
    "hallway": "Corridor",
    "bedroom": "Bedroom",
}

# Device/thing keywords -> (domain filter, name passed to HA)
THING: dict[str, tuple[list[str] | None, str | None]] = {
    "свет": (["light"], None),
    "света": (["light"], None),
    "светильник": (["light"], None),
    "светильники": (["light"], None),
    "ламп": (["light"], None),
    "подсветк": (["light"], None),
    "лампочк": (["light"], None),
    "торшер": (["light"], "торшер"),
    "люстра": (["light"], "люстра"),
    "штор": (["cover"], None),
    "шторы": (["cover"], None),
    "жалюзи": (["cover"], None),
    "пылесос": (["vacuum"], None),
    "пылесоса": (["vacuum"], None),
    "кофеварк": (["switch"], "кофеварка"),
    "кофемашин": (["switch"], "кофемашина"),
    "чайник": (["switch"], "чайник"),
    "розетк": (["switch"], None),
    "телевизор": (["media_player"], "телевизор"),
    "музык": (["media_player"], None),
    "плеер": (["media_player"], None),
    "кондиционер": (["climate"], None),
    "климат": (["climate"], None),
    "утюг": (["switch"], "утюг"),
    "обогревател": (["climate"], None),
}

RE_AREA = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ROOMS, key=len, reverse=True)) + r")\w*",
    re.IGNORECASE,
)
RE_ON = re.compile(r"\b(включи|включить|зажг|зажечь|подними|поднять)\b", re.IGNORECASE)
RE_OFF = re.compile(
    r"\b(выключи|выключить|погаси|погасить|отключи|отключить|сруби|опусти|опустить)\b",
    re.IGNORECASE,
)
RE_BRIGHTER = re.compile(r"\b(ярче|яркость|светлее|прибавь свет)\b", re.IGNORECASE)
RE_DIMMER = re.compile(
    r"\b(тусклее|темнее|приглуши|убавь|уменьши свет|потемнее)\b", re.IGNORECASE
)
RE_PCT = re.compile(r"(\d{1,3})\s*%")
RE_VACUUM_DOCK = re.compile(
    r"\b(на зарядку|в док|домой|подзарядк\w*|вернись)\b", re.IGNORECASE
)
RE_VACUUM_CLEAN = re.compile(
    r"\b(пылесос\w*|уборк\w*|прибери|убери пол|пыль)\b", re.IGNORECASE
)
RE_TIMERS = re.compile(r"\bтаймер\w*|\bнапоминан\w*|\btimer", re.IGNORECASE)
RE_BROADCAST = re.compile(
    r"^\s*(?:скажи|объяви|передай|позови)\s+(?:всем|всему дому|по дому)\b",
    re.IGNORECASE,
)
RE_COLOR = re.compile(
    r"\b(красн\w*|син\w*|зелен\w*|жёлт\w*|желт\w*|бел\w*|оранжев\w*|розов\w*|голуб\w*|фиолетов\w*)\b",
    re.IGNORECASE,
)
RE_WARM = re.compile(r"\b(тепл\w*|теплее|тёпл\w*)\b", re.IGNORECASE)
RE_COOL = re.compile(r"\b(холодн\w*|холоднее)\b", re.IGNORECASE)

COLOR_MAP = {
    "красн": "red",
    "син": "blue",
    "зелен": "green",
    "жёлт": "yellow",
    "желт": "yellow",
    "бел": "white",
    "оранжев": "orange",
    "розов": "pink",
    "голуб": "lightblue",
    "фиолетов": "purple",
}


@dataclass
class ResolvedCall:
    tool: str
    args: dict = field(default_factory=dict)
    speak_ok: str = "Готово"
    area_source: str = ""  # "explicit" | "default" | "none"
    hint: str = ""  # spoken device word ("свет") — for registry target lookup


def _find_area(text: str, stream_name: str) -> tuple[str | None, str]:
    m = RE_AREA.search(text)
    if m:
        key = m.group(1).lower()
        for stem, names in ROOMS.items():
            if key.startswith(stem):
                return names[0], "explicit"
    default = STREAM_DEFAULT_AREA.get((stream_name or "").lower())
    if default:
        return default, "default"
    return None, "none"


def _find_thing(text: str) -> tuple[list[str] | None, str | None, str, str]:
    """-> (domain, name, lowered text, matched THING stem or "")."""
    low = text.lower()
    for stem, (domain, name) in THING.items():
        if stem in low:
            return domain, name, low, stem
    return None, None, low, ""


_DEVICE_NOUN = re.compile(
    r"свет|ламп|подсветк|штор|пылесос|чайник|розетк|телевизор|музык|кофеварк|"
    r"кофемашин|утюг|кондиционер|обогревател|освещени|люстр|жалюзи|насос",
    re.IGNORECASE,
)

# Negated imperative ("не включи свет") — a wrong side-effect is the worst
# possible outcome, so escalate instead of guessing intent.
RE_NEGATION = re.compile(
    r"\b(?:не|только не|лишь не)\s+(?:пожалуйста\s+)?"
    r"(?:включи|включить|выключи|выключить|зажг|зажечь|погаси|погасить|"
    r"отключи|отключить|подними|поднять|опусти|опустить|запусти|сруби|сделай)\b",
    re.IGNORECASE,
)


def resolve_action(text: str, stream_name: str) -> ResolvedCall | None:
    """Map an imperative device command to an MCP intent tool. None = escalate."""
    t = (text or "").strip()
    if not t:
        return None
    if RE_NEGATION.search(t):
        logger.info("negated command, escalate: %r", t[:80])
        return None

    area, area_src = _find_area(t, stream_name)

    # --- Broadcast (no area) ---
    if RE_BROADCAST.search(t):
        msg = RE_BROADCAST.sub("", t, count=1).strip(" ,.!:—")
        if msg:
            return ResolvedCall(
                "assist_satellite__HassBroadcast",
                {"message": msg},
                speak_ok="Передала",
                area_source=area_src,
            )
        return None

    # --- Cancel all timers ---
    if RE_TIMERS.search(t) and RE_OFF.search(t):
        args: dict = {}
        if area:
            args["area"] = area
        return ResolvedCall(
            "intent__HassCancelAllTimers", args,
            speak_ok="Выключила все таймеры", area_source=area_src,
        )

    # --- Vacuum ---
    if RE_VACUUM_DOCK.search(t) and re.search(r"пылесос|уборк|прибери", t, re.I):
        args = {"domain": ["vacuum"]}
        if area:
            args["area"] = area
        return ResolvedCall(
            "vacuum__HassVacuumReturnToBase", args,
            speak_ok="Пылесос вернулся на зарядку", area_source=area_src,
        )
    if (
        RE_VACUUM_CLEAN.search(t)
        and re.search(r"пылесос|уборк|прибери|убери пол", t, re.I)
        and not RE_OFF.search(t)
    ):
        args = {"domain": ["vacuum"]}
        if area:
            args["area"] = area
        return ResolvedCall(
            "vacuum__HassVacuumStart", args,
            speak_ok="Запустила уборку", area_source=area_src,
        )

    # --- Light set (color / temperature / brightness) ---------------------
    # Checked BEFORE the on/off verb requirement: modulation commands use
    # their own vocabulary ("сделай ярче", "приглуши", "покрась в красный").
    thing_domain, thing_name, _low, thing_stem = _find_thing(t)
    is_light = thing_domain == ["light"] or (
        thing_domain is None and re.search(r"свет|ламп|подсветк", t, re.I)
    )
    if is_light and (
        RE_BRIGHTER.search(t) or RE_DIMMER.search(t) or RE_COLOR.search(t)
        or RE_WARM.search(t) or RE_COOL.search(t)
    ):
        args = {"domain": ["light"]}
        if area:
            args["area"] = area
        if thing_name:
            args["name"] = thing_name
        pct = RE_PCT.search(t)
        if pct:
            args["brightness"] = max(0, min(100, int(pct.group(1))))
        cm = RE_COLOR.search(t)
        if cm:
            stem = cm.group(1).lower()
            for k, v in COLOR_MAP.items():
                if stem.startswith(k):
                    args["color"] = v
                    break
        elif RE_WARM.search(t):
            args["temperature"] = 300
        elif RE_COOL.search(t):
            args["temperature"] = 500
        # Direction without a concrete value ("сделай ярче") needs the current
        # brightness to compute a step — resolver is offline-safe, so escalate
        # instead of sending a no-op HassLightSet.
        if not any(k in args for k in ("brightness", "color", "temperature")):
            return None
        speak = "Сделала"
        if "brightness" in args:
            speak = f"Яркость {args['brightness']} процентов"
        elif "color" in args:
            speak = "Сменила цвет"
        return ResolvedCall(
            "light__HassLightSet", args, speak_ok=speak, area_source=area_src,
        )

    # --- Need an action verb for plain on/off ---
    if not (RE_ON.search(t) or RE_OFF.search(t)):
        return None

    # --- Plain on/off ---
    turn_on = bool(RE_ON.search(t))
    if thing_domain is None and not _DEVICE_NOUN.search(t):
        # No recognizable device noun ("включи" alone) -> ambiguous -> escalate
        return None

    tool = "intent__HassTurnOn" if turn_on else "intent__HassTurnOff"
    args = {}
    if thing_domain:
        args["domain"] = thing_domain
    if thing_name:
        args["name"] = thing_name
    if area:
        args["area"] = area
    if not args:
        return None
    # Spoken device word: used to locate the concrete entity in the live
    # registry when the blind domain+area match fails (latin entity names).
    hint = thing_name or thing_stem
    if not hint:
        dm = _DEVICE_NOUN.search(t)
        hint = dm.group(0) if dm else ""
    return ResolvedCall(
        tool, args,
        speak_ok="Включила" if turn_on else "Выключила",
        area_source=area_src,
        hint=hint,
    )


# --- Queries ----------------------------------------------------------------

RE_TEMP = re.compile(
    r"\b(температур\w*|градус\w*)\b", re.IGNORECASE
)
RE_HUMID = re.compile(r"\b(влажност\w*|влажно)\b", re.IGNORECASE)
RE_BATT = re.compile(r"\b(заряд\w*|батаре\w*)\b", re.IGNORECASE)
RE_STATE_Q = re.compile(
    r"\b(включ[её]н|горит|работает|открыт|открыта|занят|активен|запущен|играет)\s+ли\b"
    r"|^\s*(?:а\s+)?что\s+(?:с\s+)?(?:сейчас\s+)?(?:играет|включено|идёт|идет)",
    re.IGNORECASE,
)
RE_TIME_Q = re.compile(
    r"\b(который час|какое (?:сейчас )?время|какое (?:сегодня )?число|какая дата|дата сегодня)\b",
    re.IGNORECASE,
)
RE_WEATHER_Q = re.compile(r"\bпогод\w*|улице\s+(?:жарко|холодно)", re.IGNORECASE)


@dataclass
class ResolvedQuery:
    kind: str  # "datetime" | "state" | "weather"
    args: dict = field(default_factory=dict)
    entity_hint: str = ""


def resolve_query(text: str, stream_name: str) -> ResolvedQuery | None:
    t = (text or "").strip()
    if not t:
        return None

    if RE_TIME_Q.search(t):
        return ResolvedQuery("datetime")

    if RE_WEATHER_Q.search(t) and not RE_TEMP.search(t):
        return ResolvedQuery("weather")

    area, _src = _find_area(t, stream_name)
    thing_domain, thing_name, *_ = _find_thing(t)

    if RE_TEMP.search(t):
        args: dict = {"domain": ["sensor"]}
        if area:
            args["area"] = area
        return ResolvedQuery("state", args, entity_hint="температур")
    if RE_HUMID.search(t):
        args = {"domain": ["sensor"]}
        if area:
            args["area"] = area
        return ResolvedQuery("state", args, entity_hint="влажн")
    if RE_BATT.search(t):
        args = {"domain": ["sensor"]}
        if area:
            args["area"] = area
        return ResolvedQuery("state", args, entity_hint="заряд")

    if RE_STATE_Q.search(t):
        args = {}
        if thing_domain:
            args["domain"] = thing_domain
        if thing_name:
            args["name"] = thing_name
        if area:
            args["area"] = area
        if not args:
            # "что сейчас играет" with no named device: default to the
            # media domain rather than escalating a very common query.
            if re.search(r"играет|воспроизвод|музык", t, re.I):
                args = {"domain": ["media_player"]}
            else:
                return None
        hint = thing_name or (thing_domain[0] if thing_domain else "media_player")
        return ResolvedQuery("state", args, entity_hint=hint)

    return None
