"""Semantic route classification: Ollama embeddings + calibrated cosine scoring.

PURPOSE
  First decision point of every voice turn in the L1 router (see app.py):
  map the utterance to one of the five cascade routes BEFORE anything is
  executed, so cheap deterministic handling (resolver + HA) runs for easy
  commands and the expensive L2 CodeAgent only for what needs it.

  Output contract: `Classifier.classify(text)` -> RouteDecision with the
  winning `route`, a calibrated `confidence` in [0,1], per-route `scores`
  and a non-empty `reason` whenever the verdict was ESCALATED to
  `complex_logic` (reason = why: low_confidence/low_margin/ambiguous/
  no_mcp_tool_for_request/classifier_unavailable/embed_failed/empty_text).
  app.py logs that reason into the SSE `route` event — it is the primary
  debugging aid when a command takes the slow path.

Scoring model mirrors semantic-router's Route/utterance approach (one centroid-free
max-cosine per route) but is hand-rolled because:
  * embeddings come from a self-hosted Ollama model (semantic-router's built-in
    encoders all require cloud/HF model downloads);
  * the repo has no pip on the test host, so a numpy-only module stays unit-testable;
  * thresholds are calibrated against real measurements (see config.py).

  Deterministic regex fast-paths sit ON TOP of the cosine verdict for plain
  Russian imperatives («включи свет»): an unambiguous verb pattern beats a
  doubtful cosine, except when the embedding verdict is a *confident*
  expert/complex one — «поччини и включи роутер» must stay expert.

Swap-in point: `Classifier.classify()` is the only entry — a semantic-router backend
could replace the body without touching callers.
"""

import logging
import re
from dataclasses import dataclass, field

import aiohttp
import numpy as np

import config

logger = logging.getLogger("router.classifier")

# ---------------------------------------------------------------------------
# Route definitions (utterance sets = semantic-router Route configs)
# Embedded ONCE in warmup(); a route's score is the max cosine over its own
# set, so each paraphrase widens that route's net. Sets stay deliberately
# small (10-17 lines): warmup must remain a single Ollama call, and an
# oversized example set would blur the top1/top2 margin the escalation
# rules compare against.
# ---------------------------------------------------------------------------
ROUTES: dict[str, list[str]] = {
    "easy_action": [
        "включи свет на кухне",
        "выключи свет в спальне",
        "зажги свет",
        "погаси подсветку",
        "сделай свет ярче",
        "приглуши свет",
        "убавь громкость",
        "открой шторы",
        "закрой шторы",
        "включи пылесос",
        "запусти пылесос",
        "выключи таймеры",
        "включи телевизор",
        "выключи музыку",
        "скажи всем что обед готов",
        "включи лампу в гостиной",
        "выключи все светильники",
    ],
    "easy_query": [
        "сколько градусов в спальне",
        "какая температура на кухне",
        "какая влажность в гостиной",
        "включён ли свет в коридоре",
        "горит ли свет на кухне",
        "зарядка телефона сколько процентов",
        "сколько заряда у датчика",
        "который час",
        "какая сейчас погода",
        "что сейчас играет",
        "чайник сейчас работает",
        "какое сегодня число",
    ],
    "complex_logic": [
        "если температура выше 25 то включи кондиционер",
        "каждый день в 8 утра включай свет в гостиной",
        "когда я уйду выключи всё в доме",
        "настрой сценарий доброе утро",
        "сделай так чтобы свет включался когда я захожу",
        "свяжи включение чайника с датчиком движения",
        "каждый час напоминай проверить огонь",
        "автоматизируй полив растений по расписанию",
    ],
    "expert": [
        "почему не работает роутер",
        "диагностика домашней сети",
        "почему не отвечает сервер",
        "почини nginx",
        "что случилось с пылесосом он не работает",
        "проверь логи домассистента",
        "почему не работает пылесос",
        "диагностика умного дома",
        "почему камера не показывает",
        "разберись почему нет интернета",
    ],
    "general_qa": [
        "расскажи анекдот",
        "объясни как работает квантовая запутанность",
        "как сварить кофе в турке",
        "что посоветуешь почитать",
        "напиши стих про осень",
        "почему небо голубое",
        "кто ты",
        "привет как дела",
        "что такое блокчейн",
        "пошути мне",
    ],
}

# ---------------------------------------------------------------------------
# Deterministic fast-paths (regex boost). Russian imperative device commands are
# unambiguous enough to bypass embedding doubt; resolver still validates slots.
# ---------------------------------------------------------------------------
ACTION_VERBS_ON = r"включи|включить|зажг|зажечь|подними|поднять"
ACTION_VERBS_OFF = r"выключи|выключить|погаси|погасить|отключи|отключить|сруби|опусти|опустить"

RE_ACTION = re.compile(
    rf"^\s*(?:а|и|ну)?\s*(?:пожалуйста,?\s*)?\s*(?:{ACTION_VERBS_ON}|{ACTION_VERBS_OFF})\b",
    re.IGNORECASE,
)
RE_ACTION_MID = re.compile(
    rf"\b(?:{ACTION_VERBS_ON}|{ACTION_VERBS_OFF})\b", re.IGNORECASE
)
RE_QUERY = re.compile(
    r"^\s*(?:сколько|какая|какой|каковы|который час|какое число|что)\b"
    r"|\b(?:включ[её]н|горит|работает|открыт|открыта|занят|активен|играет)\s+ли\b",
    re.IGNORECASE,
)
# Not actionable by voice: no MCP tool exists (documented limitation) => escalate.
RE_NO_TOOL = re.compile(
    r"\b(?:напомни|напоминание|напомни мне|таймер(?:ы)? на|громкость|звук)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    """Outcome of L1 classification: chosen route + scoring detail.

    `confidence` is the calibrated [0,1] margin-derived score; `scores`
    keeps the per-route cosine values for debugging; `reason` non-empty
    means the utterance was escalated (see `escalated`).
    """
    route: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    top1: float = 0.0
    top2: float = 0.0
    margin: float = 0.0
    reason: str = ""

    @property
    def escalated(self) -> bool:
        return self.reason != ""


def confidence_from_score(score: float) -> float:
    """Map cosine to [0,1]: 0.50 -> 0.0, 0.90 -> 1.0 (calibrated Sep 2026)."""
    return float(np.clip((score - 0.50) / 0.40, 0.0, 1.0))


class Embedder:
    """Async Ollama /api/embed client with batch warm-up support."""

    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.EMBED_MODEL):
        """Point the client at Ollama; the HTTP session is opened lazily."""
        self.url = url
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        """Return a live session, opening one on first use or after a close."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts` and L2-normalise the rows (cosine = dot product).

        Raises RuntimeError on a non-200 so callers can fail the warm-up
        loudly instead of classifying against a half-built matrix.
        """
        sess = await self._sess()
        async with sess.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"ollama embed {resp.status}: {body[:200]}")
            data = await resp.json()
        arr = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()


class Classifier:
    def __init__(self, embedder: Embedder | None = None,
                 conf_threshold: float = config.CONFIDENCE_THRESHOLD,
                 margin_min: float = config.MARGIN_MIN):
        """Wire the embedder and the two escalation gates.

        conf_threshold/margin_min are the calibrated floor and the
        top1-top2 spread below which a decision is considered ambiguous
        and gets escalated to L2 rather than guessed.
        """
        self.embedder = embedder or Embedder()
        self.conf_threshold = conf_threshold
        self.margin_min = margin_min
        self._matrix: np.ndarray | None = None  # (N, D) normalized utterance embeddings
        self._route_of_row: list[str] = []
        self.warmed = False

    async def warmup(self) -> bool:
        """Embed all route utterances once at startup. Failure is not fatal:
        classify() will retry lazily; on repeated failure everything escalates."""
        texts: list[str] = []
        route_of: list[str] = []
        for route, utts in ROUTES.items():
            for u in utts:
                texts.append(u)
                route_of.append(route)
        try:
            self._matrix = await self.embedder.embed(texts)
            self._route_of_row = route_of
            self.warmed = True
            logger.info(
                "classifier warmed: %d utterances, %d routes",
                len(texts), len(ROUTES),
            )
            return True
        except Exception as e:
            logger.error("classifier warmup failed: %s", e)
            self.warmed = False
            return False

    async def classify(self, text: str) -> RouteDecision:
        t = (text or "").strip()
        if not t:
            return RouteDecision("complex_logic", 0.0, reason="empty_text")

        if not self.warmed:
            await self.warmup()
        if not self.warmed or self._matrix is None:
            return RouteDecision("complex_logic", 0.0, reason="classifier_unavailable")

        try:
            v = await self.embedder.embed([t])
        except Exception as e:
            logger.error("embed failed, escalating: %s", e)
            return RouteDecision("complex_logic", 0.0, reason="embed_failed")

        cos = self._matrix @ v[0]

        # Per-route max cosine
        scores: dict[str, float] = {}
        for route in ROUTES:
            idx = [i for i, r in enumerate(self._route_of_row) if r == route]
            scores[route] = float(cos[idx].max()) if idx else 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_route, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1 - top2
        conf = confidence_from_score(top1)

        reason = ""
        final_route = top_route

        # --- Deterministic fast-paths for imperative device commands ---------
        no_tool = bool(RE_NO_TOOL.search(t))
        action_like = bool(RE_ACTION.match(t)) or bool(RE_ACTION_MID.search(t))
        query_like = bool(RE_QUERY.match(t))
        fast_path = False  # regex intent beats cosine doubt (see thresholds below)

        if action_like and not no_tool:
            # Never override a confident expert/complex verdict (e.g.
            # "почини и включи роутер" stays expert).
            if not (top_route in ("complex_logic", "expert") and conf >= self.conf_threshold):
                final_route = "easy_action"
                conf = max(conf, 0.92)
                fast_path = True
        elif query_like and not no_tool:
            if not (top_route in ("complex_logic", "expert") and conf >= self.conf_threshold):
                final_route = "easy_query"
                conf = max(conf, 0.90)
                fast_path = True

        # --- Escalation rules -----------------------------------------------
        if no_tool and final_route in ("easy_action", "easy_query"):
            return RouteDecision(
                final_route, conf, scores, top1, top2, margin,
                reason="no_mcp_tool_for_request",
            )

        if final_route in ("easy_action", "easy_query"):
            if fast_path:
                pass  # explicit regex intent: cosine margin/threshold N/A
            elif conf < self.conf_threshold:
                reason = f"low_confidence({conf:.2f}<{self.conf_threshold})"
            elif margin < self.margin_min:
                reason = f"low_margin({margin:.2f}<{self.margin_min})"
        elif conf < 0.55:
            reason = f"ambiguous({conf:.2f}<0.55)"

        if reason:
            return RouteDecision("complex_logic", conf, scores, top1, top2, margin, reason)

        return RouteDecision(final_route, round(conf, 3), scores,
                             top1, top2, margin, "")
