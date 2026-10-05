"""Intent/slot resolver for easy routes.

PURPOSE
  The deterministic brain of the L1 fast path (see app.py): after the
  classifier picked `easy_action`/`easy_query`, this module turns the Russian
  utterance into a concrete Home Assistant MCP call description *without any
  network I/O or LLM* — pure dict lookups and compiled regexes, which is why
  a voice command executes in ~0.1 s and why it is unit-testable offline
  (tests/test_router_resolution.py).

Maps a Russian voice command to an HA MCP tool call. Returns None whenever the
command is ambiguous or has no MCP counterpart — the caller then escalates to
COMPLEX (L2 CodeAgent), never a wrong side-effect.

Contracts / key behaviours
  * `resolve_action` -> ResolvedCall | None
        None  = escalate (missing verb, negation, no device noun, no slots,
                direction-only brightness, ...).
        call  = {tool, args, speak_ok, area_source, hint} — app.py executes
                it against HA and speaks `speak_ok` only after a confirmed
                success.
  * `resolve_query` -> ResolvedQuery | None
        None  = escalate; otherwise one of the kinds "datetime" / "state" /
                "weather", which app.py answers from HA, the registry or the
                open-meteo chain.
  * `unresolved_hint` -> str
        the RU "did you mean X" L2 gets when resolve_action bailed out on an
        unknown device word (STT corruption); "" on every other failure, so
        the hint can never nudge an unrelated escalation.
  * Areas: the dict sends the HA *registry display name* («Kitchen», not
    «кухня») — RU aliases exist only for some areas and «гостиная» used to
    fail with MatchFailedError INVALID_AREA (E2E regression, Sep 2026).
  * Order of the checks matters: broadcast -> timers -> vacuum -> light set
    -> plain on/off, so a sentence matching several families falls to the
    most specific one.
  * Everything here is intentionally conservative: STT typos are handled by
    stem matching («кафеварку»), but any doubt escalates to L2.
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import config
from ha_client import ordinal_digit

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
    # «во владиной комнате» (field case 03.10.2026): the Kodi's HA area is
    # named «Bedroom Vlada», so without this entry the room word matched
    # nothing, the area never left the function, and the turn was resolved as
    # «no room at all» — with four players in the house that is exactly the
    # ambiguity the fast path refuses to guess through.
    "владин": ["Bedroom Vlada", "спальня Влади"],
}

# Default area when the speaker does not name one (from the device stream name).
# Why: "выключи свет" said in the kitchen satellite must act on the kitchen —
# the physical origin of the voice is the only room information available.
# Keys are lowercased `stream_name` values from the ESP32 satellite config.
STREAM_DEFAULT_AREA: dict[str, str] = {
    "kitchen": "Kitchen",
    "livingroom": "Living Room",
    "living_room": "Living Room",
    "corridor": "Corridor",
    "hallway": "Corridor",
    "bedroom": "Bedroom",
}

# Device/thing keywords -> (domain filter, name passed to HA)
# Keys are word STEMS matched by plain substring (`stem in lowered_text`), so
# every inflected form (лампочку, шторы, пылесосом) hits the same entry and
# STT typos with the stem intact still match. Value semantics:
#   domain  -> forwarded as args["domain"] (None = let the matcher decide);
#   name    -> args["name"], the exact device word for the intent matcher,
#              or None when the stem alone (свет, штор) is enough.
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
    "робот": (["vacuum"], None),  # «робот/робота» = the same Valetudo vacuum
    "кофеварк": (["switch"], "кофеварка"),
    "кафеварк": (["switch"], "кофеварка"),  # STT hears «кафеварку» (dropped «о»)
    "кофемашин": (["switch"], "кофемашина"),
    "чайник": (["switch"], "чайник"),
    "розетк": (["switch"], None),
    "телевизор": (["media_player"], "телевизор"),
    "музык": (["media_player"], None),
    "плеер": (["media_player"], None),
    # «коди» is the word the user actually uses for the four Kodi boxes
    # (media_player.le_vlada / le_zal_2 / le_spalnya / le_kitchen), and it
    # matched nothing before 03.10.2026 — «пауза коди во владиной комнате»
    # found no device at all and L2 fell back to the robot from the dialogue
    # history. `name` stays None: the HA matcher does not know the word
    # «коди», the registry resolver does (hint -> _HINT_LAT -> «le»).
    "коди": (["media_player"], None),
    "kodi": (["media_player"], None),
    "медиаплеер": (["media_player"], None),
    "колонк": (["media_player"], None),
    "кондиционер": (["climate"], None),
    "климат": (["climate"], None),
    "утюг": (["switch"], "утюг"),
    "обогревател": (["climate"], None),
}

# --- Compiled matchers -------------------------------------------------------
# All are case-insensitive and match STEMS rather than whole words, because
# Russian inflection («включи», «включить», «включён») and STT typos both
# change word endings. RE_ON/RE_OFF deliberately mirror
# classifier.ACTION_VERBS_ON/ACTION_VERBS_OFF: the classifier decides the
# ROUTE from those verbs, this module decides the TOOL — if the two lists
# drifted apart, a command would be routed easy_action and then escalated
# as unresolvable (a silent loss of the fast path).
RE_AREA = re.compile(
    # Longest stem first: Python alternation is ordered, so a shorter stem
    # can never shadow a longer alternative sharing the same start position.
    # The trailing \w* swallows the rest of the word (кухня/кухне/кухонный).
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
# Percent is captured, not just matched: the digit (clamped to 0..100 when
# stored) becomes args["brightness"] — see the light-set branch below.
RE_PCT = re.compile(r"(\d{1,3})\s*%")
# Vacuum "go back to the dock" vocabulary. resolve_action additionally
# requires an explicit vacuum word, so a stray «вернись»/«домой» alone can
# never trigger a device call.
RE_VACUUM_DOCK = re.compile(
    r"\b(на зарядку|в док|домой|подзарядк\w*|вернись)\b", re.IGNORECASE
)
RE_VACUUM_CLEAN = re.compile(
    r"\b(пылесос\w*|уборк\w*|прибери|убери пол|пыль)\b", re.IGNORECASE
)
# Timer/reminder vocabulary: meaningful only together with RE_OFF — that
# combination is what builds the cancel-all intent («выключи таймеры»).
RE_TIMERS = re.compile(r"\bтаймер\w*|\bнапоминан\w*|\btimer", re.IGNORECASE)
# Anchored at ^: only a LEADING announcement verb + audience is a broadcast.
# resolve_action strips this prefix with sub(count=1) and speaks the rest —
# «скажи всем что обед готов» -> message «что обед готов».
RE_BROADCAST = re.compile(
    r"^\s*(?:скажи|объяви|передай|позови)\s+(?:всем|всему дому|по дому)\b",
    re.IGNORECASE,
)
# Light-modulation matchers (checked BEFORE plain on/off — these commands
# have their own vocabulary and no on/off verb): RE_COLOR captures a hue
# stem (красн -> красный/красная), RE_WARM/RE_COOL cover colour TEMPERATURE
# («теплее/холоднее» -> 300 K / 500 K, i.e. Kelvin for a light, never a hue).
RE_COLOR = re.compile(
    r"\b(красн\w*|син\w*|зелен\w*|жёлт\w*|желт\w*|бел\w*|оранжев\w*|розов\w*|голуб\w*|фиолетов\w*)\b",
    re.IGNORECASE,
)
RE_WARM = re.compile(r"\b(тепл\w*|теплее|тёпл\w*)\b", re.IGNORECASE)
RE_COOL = re.compile(r"\b(холодн\w*|холоднее)\b", re.IGNORECASE)

# Russian colour stem -> HA colour name. Looked up with stem.startswith on
# the word RE_COLOR captured, so «красную»/«красная»/«красный» all -> "red"
# (the captured group is matched case-insensitively and lowered first).
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
    """One resolved easy_action — the complete input of app._execute_action.

    tool       -> HA MCP tool name (intent__*, light__*, vacuum__*,
                  assist_satellite__*).
    args       -> matcher slots (domain/name/area/brightness/color/
                  temperature); `area` holds the HA registry DISPLAY name
                  («Kitchen»), never the raw RU word.
    speak_ok   -> phrase to speak, but only after HA confirms the call.
    area_source-> "explicit" (named in the utterance) | "default" (taken
                  from the satellite stream) | "none" — lets the caller tell
                  a room the user said from one the router assumed.
    hint       -> spoken device stem ("свет"), used to locate the concrete
                  entity in the live registry when the blind domain+area
                  match fails (latin entity names, see ha_client).
    """

    tool: str
    args: dict = field(default_factory=dict)
    speak_ok: str = "Готово"
    area_source: str = ""  # "explicit" | "default" | "none"
    hint: str = ""  # spoken device word ("свет") — for registry target lookup


def _find_area(text: str, stream_name: str) -> tuple[str | None, str]:
    """(HA area display name or None, source) for the utterance.

    Priority: a room word in the text > the satellite's default room >
    no area at all. Returns the registry display name from ROOMS[names[0]],
    which is what the HA intent matcher accepts (RU aliases are incomplete).
    """
    m = RE_AREA.search(text)
    if m:
        key = m.group(1).lower()
        # RE_AREA may have appended \w* to the stem («кухонный»); startswith
        # maps the match back to the ROOMS key that produced it.
        for stem, names in ROOMS.items():
            if key.startswith(stem):
                return names[0], "explicit"
    default = STREAM_DEFAULT_AREA.get((stream_name or "").lower())
    if default:
        return default, "default"
    return None, "none"


def _find_thing(text: str) -> tuple[list[str] | None, str | None, str, str]:
    """-> (domain, name, lowered text, matched THING stem or "").

    First THING stem contained in the (lowered) text wins — dict order is
    the priority, and callers reuse the returned lowered text for their own
    regex checks instead of re-lowering.
    """
    low = text.lower()
    for stem, (domain, name) in THING.items():
        if stem in low:
            return domain, name, low, stem
    return None, None, low, ""


# Fallback noun list for the plain on/off branch: broader than THING (adds
# «освещение», «люстра», «насос» ...) and used for two things — refusing a
# bare «включи» with no device at all (escalate), and deriving the `hint`
# when THING had no stem to offer.
_DEVICE_NOUN = re.compile(
    r"свет|ламп|подсветк|штор|пылесос|чайник|розетк|телевизор|музык|кофеварк|"
    r"кафеварк|кофемашин|утюг|кондиционер|обогревател|освещени|люстр|жалюзи|насос|"
    r"коди|медиаплеер|плеер|колонк|кино|фильм|сериал|сери[юяи]|эпизод",
    re.IGNORECASE,
)

# --- Media transport -------------------------------------------------------
# HA's MCP server exposes TEN tools and not one of them is a media intent
# (tools/list, read live 03.10.2026), which is why «поставь его на паузу»
# ended in «не удалось»: the model had to invent intent__HassMediaPause.
# The SERVICES do exist (`media_player.media_pause`), so a media command is
# resolved here into a REST call instead: tool = "media__" + service, and
# app._execute_action sends it to /api/services. This is the fast path the
# same command used to lose 7 s and a turn to reach L2 for.
#
# Keys are HA service names — the same names smolagents-worker/ha_match.py
# accepts in media_control(action=…), and tests/test_hint_sync.py fails if
# the two tables drift.
MEDIA_SERVICES: dict[str, str] = {
    "media_pause": "media_pause",
    "media_play": "media_play",
    "media_play_pause": "media_play_pause",
    "media_stop": "media_stop",
    "media_next_track": "media_next_track",
    "media_previous_track": "media_previous_track",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "volume_set": "volume_set",
    "volume_mute": "volume_mute",
}

# (regex, service, exclusive) triples, checked in order — the first match
# wins, so the specific phrases precede the loose ones. `exclusive` marks a
# word that can ONLY mean transport («пауза», «громче», «следующий трек»):
# it resolves even with no device noun, because «поставь ЕГО на паузу» names
# its target with a pronoun the registry can answer. Non-exclusive words
# («приглуши», «заглуши») are shared with the light/TV families and need an
# object.
RE_MEDIA: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"\bпауз\w*|\bприостанов\w*|\bзамороз\w*", re.IGNORECASE),
     "media_pause", True),
    (re.compile(r"\bпродолж\w*|\bвозобнов\w*|\bдоигр\w*|сними\s+с\s+паузы",
                re.IGNORECASE), "media_play", True),
    # «трек/песня» only, NOT «серию/эпизод»: those are LIBRARY items, and a
    # transport next_track on the wrong box is a wrong side effect — «включай
    # следующую серию "Темного зеркала" в гостиной» must go to L2's
    # media_search + media_play, with the room from the escalation context.
    (re.compile(r"\bследующ\w*\s+(трек|песн|композиц)|\bдальше\b|"
                r"\bпереключ\w*\s+трек", re.IGNORECASE),
     "media_next_track", True),
    (re.compile(r"\bпредыдущ\w*\s+(трек|песн|композиц)|\bназад\b",
                re.IGNORECASE),
     "media_previous_track", True),
    (re.compile(r"\bгромче\b|\bприбав\w*\s+(громкост|звук)", re.IGNORECASE),
     "volume_up", True),
    (re.compile(r"\bтише\b|\bубав\w*\s+(громкост|звук)|"
                r"\bуменьш\w*\s+(громкост|звук)", re.IGNORECASE),
     "volume_down", True),
    # «приглуши» is split off: on its own it is the HOUSE'S dimming verb
    # (RE_DIMMER handles it for lights), so it only means a player when an
    # object is named — hence exclusive=False.
    (re.compile(r"\bприглуш\w*", re.IGNORECASE), "volume_down", False),
    (re.compile(r"\bзаглуш\w*|\bвыключ\w*\s+звук|\bзвук\s+выключ|"
                r"\bmute", re.IGNORECASE), "volume_mute", False),
    (re.compile(r"\bвключ\w*\s+звук|\bразглуш\w*|\bзвук\s+включ|"
                r"\bunmute", re.IGNORECASE), "volume_mute", False),
    (re.compile(r"\bгромкост\w*|\bгромк\w*|\bvolume", re.IGNORECASE),
     "volume_set", True),
    (re.compile(r"\b(?:останов|стоп)\w*\s+(?:плеер|воспроизвод|музык|фильм|видео)",
                re.IGNORECASE), "media_stop", True),
)

# Any word that says «this sentence is about a player» — enough to keep the
# light/vacuum branches from claiming it and to accept a shared verb. The
# CONTENT words («серию», «эпизод») belong here because that is how the user
# asks for it: «Включай следующую серию "Темного зеркала в гостиной» cost an
# 11 s L2 round trip before anyone noticed the room held a player at all.
RE_MEDIA_NOUN = re.compile(
    r"коди|kodi|медиаплеер|медиа\s+плеер|плеер|колонк|телевизор|музык|"
    r"кино|фильм|сериал|сери[юяи]|эпизод|часть|трек|песн|воспроизвод|"
    r"передач[уаи]|ролик|клип",
    re.IGNORECASE,
)

# Device words that may travel as the registry hint. «трек»/«кино» are left
# out on purpose: they name WHAT is playing, not WHICH box, and a hint that
# matches no entity only narrows the pool wrongly.
RE_MEDIA_HINT = re.compile(
    r"коди|kodi|медиаплеер|медиа\s+плеер|плеер|колонк|телевизор", re.IGNORECASE,
)

# A volume level. Two spellings matter: «громкость 40» (the number sits right
# after the word and carries NO unit, which RE_PCT — the light branch's, «40%»
# only — cannot see) and «40 процентов».
RE_MEDIA_PCT = re.compile(
    r"\b(?:громкост\w*|громк\w*|уровень|volume)\D{0,12}?(\d{1,3})"
    r"|(\d{1,3})\s*(?:%|процент\w*)",
    re.IGNORECASE,
)

# What «включи/выключи <плеер>» means for a media_player. NOT HassTurnOn/Off:
# field check 03.10.2026 18:21, «Ну так найди её и включи» — the intent answered
# MatchFailedReason.INVALID_AREA for the RU room and then
# MatchFailedReason.ASSISTANT for `name='LE-zal'`, because the Kodi entities are
# NOT exposed to the voice assistant and the MCP server has no media intent at
# all. The transport services work regardless of Assist exposure, so «включи
# коди» is media_play and «выключи коди» is media_stop. Powering the BOX off is
# deliberately NOT `homeassistant.turn_off`: the box would sleep and no voice
# command could wake it again.
MEDIA_POWER_ON = "media_play"
MEDIA_POWER_OFF = "media_stop"

# Phrase to speak after a confirmed call. Kept short and TTS-shaped; the
# idempotent cases («уже на паузе») are decided from the live state in
# app._execute_media.
MEDIA_SPEAK: dict[str, str] = {
    "media_pause": "Поставила на паузу",
    "media_play": "Продолжаю",
    "media_play_pause": "Переключила",
    "media_stop": "Остановила",
    "media_next_track": "Следующий трек",
    "media_previous_track": "Предыдущий трек",
    "volume_up": "Сделала громче",
    "volume_down": "Сделала тише",
    "volume_set": "Поставила громкость",
    "volume_mute": "Выключила звук",
}

# Negated imperative ("не включи свет") — a wrong side-effect is the worst
# possible outcome, so escalate instead of guessing intent.
RE_NEGATION = re.compile(
    r"\b(?:не|только не|лишь не)\s+(?:пожалуйста\s+)?"
    r"(?:включи|включить|выключи|выключить|зажг|зажечь|погаси|погасить|"
    r"отключи|отключить|подними|поднять|опусти|опустить|запусти|сруби|сделай)\b",
    re.IGNORECASE,
)


def area_of(text: str, stream_name: str) -> tuple[str | None, str]:
    """Public `_find_area`: (HA area display name or None, source).

    app.py needs it to hand L2 the CANONICAL room name on every escalation:
    «гостиная» is not an HA area (it is `Living Room`), and passing the RU word
    straight to an intent answered MatchFailedReason.INVALID_AREA and cost the
    model a step it then spent guessing device names instead (field case
    03.10.2026 18:21).
    """
    return _find_area(text or "", stream_name)


def resolve_action(text: str, stream_name: str) -> ResolvedCall | None:
    """Map an imperative device command to an MCP intent tool. None = escalate.

    Returns a ResolvedCall only when EVERY required slot is known; returns
    None (caller escalates to L2) for an empty/negated command, a missing
    action verb, an unrecognizable device, an empty args dict, or a
    direction-only brightness request that would need the current value.

    Check order is significant: broadcast -> timers -> vacuum -> light
    modulation -> plain on/off, so a sentence matching several families
    resolves to the most specific intent.
    """
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
        # Field regression 25.09.2026 («уберется на кухне»): in HassVacuumStart
        # the area slot filters by the vacuum's LOCATION, and the robot is
        # assigned to no room in HA -> MatchFailedReason.AREA every time.
        # A room NAMED in the utterance is the cleaning target — that is
        # HassVacuumCleanArea, where area is a service parameter and the
        # entity match does not depend on the robot's location. The stream's
        # default area is only a speaker-location hint (not the user's
        # cleaning target), so it keeps the old Start+area behaviour.
        if area and area_src == "explicit":
            return ResolvedCall(
                "vacuum__HassVacuumCleanArea", {"area": area},
                speak_ok="Запустила уборку", area_source=area_src,
            )
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

    # --- Media transport (pause / tracks / volume) --------------------------
    # BEFORE the on/off branch on purpose: «выключи звук» matches RE_OFF, and
    # HassTurnOff cannot mute anything — the transport services must win.
    media_noun = bool(RE_MEDIA_NOUN.search(t))
    media_service, media_exclusive = "", False
    for pattern, service, exclusive in RE_MEDIA:
        if pattern.search(t):
            media_service, media_exclusive = service, exclusive
            break
    # «включи/выключи <плеер>» joins them: HassTurnOn/HassTurnOff CANNOT
    # control these entities (no media intent in MCP, and the Kodis are not
    # exposed to Assist — both proven in the field on 03.10.2026 18:21), so
    # it is media_play/media_stop. Only a media NOUN qualifies: a bare
    # «включи» stays with the light/vacuum intents.
    if (
        not media_service
        and media_noun
        and thing_domain == ["media_player"]
        and (RE_ON.search(t) or RE_OFF.search(t))
    ):
        media_service = MEDIA_POWER_ON if RE_ON.search(t) else MEDIA_POWER_OFF
    # A device word from another family wins over any media reading:
    # «включи свет и поставь музыку» is not a media-only command, and half a
    # command is an escalation, never a partial side effect.
    if media_service and thing_domain and thing_domain != ["media_player"]:
        return None
    if media_service and not (media_exclusive or media_noun or thing_stem):
        # A shared verb with no object («приглуши», «заглуши», «стоп») — in
        # this house those are light/TV words; refuse to guess the device.
        if not re.search(r"громкост|звук", t, re.IGNORECASE):
            return None
    if media_service:
        if area_src == "default" and not media_noun and not thing_stem:
            # «громче» said in the kitchen satellite with no device word: the
            # default area is a speaker LOCATION, not a claim about which
            # player the user means.
            area, area_src = None, "none"
        args: dict = {}
        if area:
            args["area"] = area
        hint = thing_name or thing_stem
        if not hint:
            m = RE_MEDIA_HINT.search(t)
            hint = m.group(0) if m else ""
        service_data: dict = {}
        speak = MEDIA_SPEAK[media_service]
        if media_service == "volume_set":
            pct = RE_MEDIA_PCT.search(t)
            if not pct:
                return None  # «сделай громче»-less «поставь громкость»: escalate
            level = int(pct.group(1) or pct.group(2))
            service_data["volume_level"] = round(min(100, level) / 100.0, 2)
            speak = f"Громкость {level} процентов"
        elif media_service == "volume_mute":
            # RE_MEDIA maps the sound-on case to volume_mute as well; the flag
            # is what separates them, and «выключи звук» must not unmute.
            service_data["is_volume_muted"] = not bool(
                re.search(r"включ\w*\s+звук|разглуш|unmute", t, re.IGNORECASE))
            speak = ("Выключила звук" if service_data["is_volume_muted"]
                     else "Включила звук")
        args["service_data"] = service_data
        return ResolvedCall(
            f"media__{media_service}", args,
            speak_ok=speak, area_source=area_src, hint=hint,
        )

    # --- Need an action verb for plain on/off ---
    if not (RE_ON.search(t) or RE_OFF.search(t)):
        return None

    # --- Plain on/off ---
    # Tie-break: when both verb families appear in one sentence, ON wins.
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
        # No slot at all (device known only from _DEVICE_NOUN, no room, no
        # default area): a filterless call would toggle an arbitrary entity.
        return None
    # Spoken device word: used to locate the concrete entity in the live
    # registry when the blind domain+area match fails (latin entity names).
    hint = thing_name or thing_stem
    if not hint:
        dm = _DEVICE_NOUN.search(t)
        hint = dm.group(0) if dm else ""
    # «в первом коридоре»: the ordinal travels INSIDE the hint, because the
    # slots are HA matcher slots and a digit is not one of them. Dropped
    # before 02.10.2026, so «первый» and «второй» both moved corridor1 AND
    # corridor2 — find_action_targets reads the digit back and keeps only
    # the relay the user named.
    digit = ordinal_digit(t)
    if hint and digit:
        hint = f"{hint} {digit}"
    return ResolvedCall(
        tool, args,
        speak_ok="Включила" if turn_on else "Выключила",
        area_source=area_src,
        hint=hint,
    )


# --- Escalation hint: near-miss device words --------------------------------
# Field case 28.09.2026 («Выключи кашеварку»): one corrupted word made
# resolve_action bail out, and L2 received the bare utterance — it guessed a
# name, HA answered MatchFailedError, the honesty veto corrected the lie and
# the turn was lost (1 of 5 turns of that dialogue). resolve_action itself
# must stay conservative (a fuzzy match may NOT pick a device on its own), so
# the hint below does the opposite: it TELLS L2 what the unknown word most
# probably means and lets the model — which sees the whole utterance plus the
# dialogue history — decide whether to use it.
#
# Narrow by construction, all four guards must pass:
#   * a command verb is present (a question/state gets no nudge to act);
#   * _DEVICE_NOUN does not match — i.e. the exact branch that made
#     resolve_action return None ("no recognizable device noun");
#   * no negation («не выключи кофеварку» must never be nudged toward an
#     action the user explicitly refused);
#   * no candidate that the dictionary already knows (then the fast path
#     failed for a verb/slot reason, not because the device is unknown).
_HINT_THRESHOLD = 0.70  # difflib ratio; «кашеварк» vs «кофеварк» = 0.75

# Verbs, pronouns and fillers — words that can never name a device. Matched
# against a whole token, so «выключи» is dropped while «кашеварку» survives.
_RE_HINT_STOP = re.compile(
    r"^(?:включ\w*|выключ\w*|зажг\w*|зажечь|погас\w*|отключ\w*|подними|поднять|"
    r"опусти|опустить|сдела\w*|постав\w*|убери|убрать|запусти|запустить|"
    r"пожалуйста|давай|просто|будто|может|чтоб|"
    r"это|эта|этот|эти|её|ее|он|она|они|оно|тот|та|те|то|"
    r"вс[её]|весь|вся|мой|моя|мо[её]|наш|наша|сам|сама|"
    r"такой|такая|здесь|там|сейчас|пока|потом|ну|же|бы|ли|что|как)$",
    re.IGNORECASE,
)

# Every word the hint may match against -> the name to hand to ha_action.
# Both spellings matter: the stem («кофеварк») catches inflected forms, the
# exact name («кофеварка») is what makes a heavily corrupted word still line
# up («криварка» scores 0.706 against the name but only 0.667 against the
# stem — i.e. it would lose its hint without this).
_HINT_VOCAB: list[tuple[str, str]] = list(
    dict.fromkeys(
        pair
        for stem, (_domain, name) in THING.items()
        for pair in ((stem, name or stem), ((name or stem), name or stem))
    )
)
for _extra in ("освещени", "люстр", "насос"):
    _HINT_VOCAB.append((_extra, _extra))


def _hint_candidates(text: str) -> list[str]:
    """Words of the utterance that may still be a (misheard) device name.

    Filters: >= 5 chars (short words match stems too easily), stop words,
    action verbs (RE_ON/RE_OFF) and room words (RE_AREA) — all of them were
    already understood by the fast path, so they cannot be the unknown noun.
    """
    out: list[str] = []
    for w in re.findall(r"[а-яё]+", text.lower()):
        if len(w) < 5 or _RE_HINT_STOP.match(w):
            continue
        if RE_ON.match(w) or RE_OFF.match(w) or RE_AREA.match(w):
            continue
        out.append(w)
    return out


def _hint_norm(word: str) -> str:
    """Lowercase + ё→е + ONE case ending dropped («кашеварку» -> «кашеварк»).

    One character only: Russian endings can be longer («-ому», «-ями»), but
    stripping more would turn a device into a different device — and the
    fuzzy step only needs the noun stem to line up.
    """
    w = word.lower().replace("ё", "е")
    if w.endswith("ь"):
        return w[:-1]
    if len(w) > 4 and w[-1] in "аяыеиоуюе":
        return w[:-1]
    return w


def _hint_sim(a: str, b: str) -> float:
    """Best difflib ratio of the raw and the de-inflected spellings."""
    best = SequenceMatcher(None, a, b).ratio()
    na, nb = _hint_norm(a), _hint_norm(b)
    if (na, nb) != (a, b):
        best = max(best, SequenceMatcher(None, na, nb).ratio())
    return best


def device_hint_from(text: str, max_words: int = 3) -> str:
    """The DEVICE a previous turn talked about, with its verb removed.

    Exists because a follow-up carries the action but not the target: «включи
    свет» then «а теперь выключи» leaves `resolve_action` with a verb and nothing
    to apply it to, so it returns None and the turn escalates. Measured 05.10.2026:
    the user said exactly that and the router answered `resolver_ambiguous`
    after 6 s of L2 — the dialogue looks broken because the room does not
    remember what it was just talking about.

    Only the noun survives. Appending the whole previous text instead is unsafe
    and was measured to be: `resolve_action("а теперь выключи включи свет")`
    returns **HassTurnOn**, because the resolver scans for any verb and finds
    «включи» in the borrowed text. The verb is the part we must NOT inherit; the
    device is the part we must.

    Returns "" when there is nothing noun-shaped to take, which is the case for
    «включи» on its own and for a turn that named no device.
    """
    t = (text or "").strip().lower()
    if not t:
        return ""
    # Drop a leading action/filler prefix: «а теперь выключи свет» -> «свет».
    # Repeated, because the fillers stack: «а теперь выключи» has two of them and
    # a single pass stripped only «а», leaving «теперь» as the device.
    t = re.sub(
        r"^(?:\s*(?:и|а|ну|так|теперь|да|вот|"
        r"пожалуйста|please)\b[\s,]*)+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    for rx in (RE_ON, RE_OFF, RE_BRIGHTER, RE_DIMMER):
        t = rx.sub(" ", t, count=1)
    t = re.sub(r"\b(?:в|во|на|у|для|мне|мне\s+пожалуйста|все|всё|там|тут)\b", " ", t)
    t = re.sub(r"[^\w\s-]", " ", t).strip()
    words = [w for w in t.split() if w]
    if not words or len(words) > max_words:
        return ""
    return " ".join(words)


def unresolved_hint(text: str) -> str:
    """RU hint for L2 when the fast path could not name the device, else "".

    Returned text is a suggestion, never an instruction: it names the
    near-miss and says what to do ONLY IF that is what the user meant, so a
    wrong guess costs L2 one wasted call instead of a wrong side effect.
    Pure and offline (no registry access) — see tests/test_router_resolution.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if not (RE_ON.search(t) or RE_OFF.search(t)):
        return ""  # not an on/off command: no device is being requested
    if _DEVICE_NOUN.search(t):
        return ""  # the dictionary knows a device word: failed for another reason
    if RE_NEGATION.search(t):
        return ""  # user refused the action: never nudge toward it
    cands = _hint_candidates(t)
    if not cands:
        return ""  # pronoun/room only («выключи её») — history, not a name guess
    for c in cands:
        if any(needle in c for needle, _name in _HINT_VOCAB):
            return ""  # the dictionary already knows this word
    best_cand, best_name, best_r = "", "", _HINT_THRESHOLD
    for c in cands:
        for needle, name in _HINT_VOCAB:
            r = _hint_sim(c, needle)
            if r > best_r:
                best_cand, best_name, best_r = c, name, r
    if not best_cand:
        return ""
    return (
        f"Слово «{best_cand}» не распознано — вероятно, STT его исказил. "
        f"Похожее устройство: «{best_name}». Если это оно, вызывай ha_action "
        f"с точным именем «{best_name}»."
    )


# --- Queries ----------------------------------------------------------------

# Query regexes: they pick the ANSWER SOURCE, not a tool. The captured kind
# decides which backend app.py talks to (HA datetime tool / open-meteo /
# live registry), and `entity_hint` is always a RU stem — ha_client._HINT_LAT
# expands it to the latin fragments that exist in entity_ids.
RE_TEMP = re.compile(
    r"\b(температур\w*|градус\w*)\b", re.IGNORECASE
)
RE_HUMID = re.compile(r"\b(влажност\w*|влажно)\b", re.IGNORECASE)
RE_BATT = re.compile(r"\b(заряд\w*|батаре\w*)\b", re.IGNORECASE)
# Two shapes: the yes/no form («включён ли свет») and the what-is-playing
# form («что сейчас играет») — both resolve to a registry state read.
RE_STATE_Q = re.compile(
    r"\b(включ[её]н|горит|работает|открыт|открыта|занят|активен|запущен|играет)\s+ли\b"
    r"|^\s*(?:а\s+)?что\s+(?:с\s+)?(?:сейчас\s+)?(?:играет|включено|идёт|идет)",
    re.IGNORECASE,
)
# Status phrasing about a NAMED device: «что там с нашим пылесосом?»,
# «что у нас с роботом?», «что с чайником», «чего с роботом?», «как пылесос?».
# The optional fillers (там/у нас/у тебя/вообще/в целом) are a chain, not a
# single slot: «что у нас с роботом?» (29.09.2026, 14:17) matched NONE of the
# earlier spellings, escalated to L2 and got a fabricated answer — 0.3 s of
# deterministic registry lookup instead. The second branch requires the
# question mark: STT keeps it, and without it «как включить свет» would read
# as a status question. Only ever consulted together with a THING match (see
# resolve_query), so «как дела?»/«привет» still escalate.
RE_STATUS_Q = re.compile(
    r"\b(?:что|чего)\s+(?:там\s+|нового\s+|у\s+нас\s+|у\s+тебя\s+|"
    r"вообще\s+|в\s+целом\s+)*(?:с|со)\b"
    r"|\bкак\s+(?:наш\w*|мой\w*|он|она|оно|сейчас)?\s*[\wа-яё-]+\s*\?",
    re.IGNORECASE,
)
RE_TIME_Q = re.compile(
    r"\b(который час|какое (?:сейчас )?время|какое (?:сегодня )?число|какая дата|дата сегодня)\b",
    re.IGNORECASE,
)
RE_WEATHER_Q = re.compile(r"\bпогод\w*|улице\s+(?:жарко|холодно)", re.IGNORECASE)


@dataclass
class ResolvedQuery:
    """One resolved easy_query — the input of the app.py easy_query branch.

    kind       -> "datetime" (HA llm__GetDateTime), "weather" (open-meteo
                  hybrid chain) or "state" (live registry lookup).
    args       -> optional matcher slots for the state read: domain/name/
                  area, with `area` again the HA registry display name, plus
                  `label` — a RU device word («пылесос») that describe_entity
                  speaks instead of the latin friendly name.
    entity_hint-> RU stem handed to ha_client.find_entity() for the
                  bilingual (RU stem -> latin entity) match.
    """

    kind: str  # "datetime" | "state" | "weather"
    args: dict = field(default_factory=dict)
    entity_hint: str = ""


def resolve_query(text: str, stream_name: str) -> ResolvedQuery | None:
    """Classify an easy_query utterance -> ResolvedQuery, or None = escalate.

    Order matters: datetime first (unambiguous phrasing), then weather —
    but only when the question carries NO temperature word, otherwise a
    mixed «градусы и погода» question would be answered from the forecast
    instead of the sensor — then the sensor/state families.
    Returns None for an empty input or a phrasing none of the regexes
    cover; app.py then escalates to complex_logic.
    """
    t = (text or "").strip()
    if not t:
        return None

    # datetime: HA-side clock, no slots to fill.
    if RE_TIME_Q.search(t):
        return ResolvedQuery("datetime")

    # Weather only if no temperature word (see docstring).
    if RE_WEATHER_Q.search(t) and not RE_TEMP.search(t):
        return ResolvedQuery("weather")

    # The area *source* is irrelevant for queries (nothing is toggled).
    area, _src = _find_area(t, stream_name)
    thing_domain, thing_name, _low, thing_stem = _find_thing(t)

    # Sensor reads: domain=sensor (+ room when known); the RU stem travels
    # in entity_hint and ha_client.find_entity() does the bilingual lookup.
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
        # «сколько заряда у робота» (29.09.2026): the METRIC alone matches
        # every battery in the house and find_entity returns the first one
        # in registry order — the phone («SM-A546E Battery level: 33
        # процентов» answered for the robot). The device word travels in
        # `device`, and find_entity() then competes only inside it.
        # `label` says WHAT is being read: without it the head of the
        # phrase falls back to the latin friendly name.
        if thing_stem or thing_name:
            args["device"] = thing_name or thing_stem
        args["label"] = thing_name or thing_stem or "заряд"
        # find_entity() returns None when the named device has no such
        # reading (it refuses to borrow another device's number — «Чайник:
        # 33 процентов» would be that borrowing). The generic «не нашла
        # такого устройства» would be wrong too: the device is known, the
        # READING is what does not exist, so say exactly that.
        args["missing"] = (
            "Данных о заряде этого устройства в Home Assistant нет."
            if (thing_stem or thing_name)
            else "Данных о заряде в Home Assistant нет."
        )
        return ResolvedQuery("state", args, entity_hint="заряд")

    # Status question about a NAMED device («что там с нашим пылесосом?»):
    # THING already resolved the noun to a domain, so this is a registry
    # state read in ~1 s instead of an L2 escalation. Field case 29.09.2026:
    # the phrasing fell through to None, the free model was asked instead
    # and invented «работает, заряда 45 %» while HA reported `docked` (and
    # has no battery attribute for that vacuum at all) — 20 s of latency for
    # a lookup this branch does deterministically.
    if thing_domain and RE_STATUS_Q.search(t):
        args = {"domain": thing_domain}
        if thing_name:
            args["name"] = thing_name
        if area:
            args["area"] = area
        # RU stem («пылесос») rather than the domain: ha_client._HINT_LAT
        # expands it to vacuum/roborock/robot, and describe_entity() speaks
        # it back instead of the latin «Roborock Robot».
        args["label"] = thing_name or thing_stem
        return ResolvedQuery(
            "state", args, entity_hint=thing_stem or thing_name or thing_domain[0]
        )

    if RE_STATE_Q.search(t):
        args = {}
        if thing_domain:
            args["domain"] = thing_domain
        if thing_name:
            args["name"] = thing_name
        if area:
            args["area"] = area
        # A playback question with a ROOM and no device word is about the
        # players of that room: the media default used to sit behind
        # `if not args`, so «что сейчас играет в гостиной» carried the area
        # alone and answered «Не нашла такого устройства» about a TV that was
        # playing (field check 03.10.2026).
        if (
            not thing_domain
            and re.search(r"играет|воспроизвод|музык|фильм|сериал|показ", t, re.I)
        ):
            args["domain"] = ["media_player"]
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
