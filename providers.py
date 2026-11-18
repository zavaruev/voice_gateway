"""A small provider chain with a circuit breaker.

The house has three TTS engines, an LLM cascade and a Whisper endpoint, and
every one of them has failed in production. The failure used to be either
silence or a manual POST to switch engines at 3 a.m. A chain does two things:

  1. tries the next provider when one fails or times out;
  2. LOGS THE DOWNGRADE with the provider name and the reason.

Rule from this codebase: a fallback that cannot be seen is worse than no
fallback — it turns a broken service into a mystery. Every skip, failure and
recovery is one line in the log; `status()` names the providers currently in
cooldown so the next investigation starts with a fact.

The providers are async callables. Nothing here knows what TTS or STT is, so
it is unit-testable with functions that just raise.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

logger = logging.getLogger("providers")

ProviderFn = Callable[..., Awaitable]


@dataclass
class Provider:
    name: str
    fn: ProviderFn
    cooldown_s: float = 120.0
    max_failures: int = 2
    failures: int = 0
    last_failure: float = 0.0
    last_error: str = ""
    calls: int = 0

    def available(self, now: float) -> bool:
        if self.failures < self.max_failures:
            return True
        return now - self.last_failure >= self.cooldown_s

    def record_success(self) -> None:
        self.failures = 0
        self.last_error = ""

    def record_failure(self, exc: BaseException, now: float) -> None:
        self.failures += 1
        self.last_failure = now
        self.last_error = f"{type(exc).__name__}: {exc}"[:160]


@dataclass
class CallResult:
    value: object
    provider: str
    downgrades: list[str] = field(default_factory=list)


class ProviderChain:
    """Try providers in order; remember the ones that keep failing.

    A provider is only skipped after `max_failures` consecutive failures, and
    is retried after `cooldown_s` — so a single transient blip does not demote
    an engine for the rest of the day, and a permanently dead one is not tried
    on every single turn.
    """

    def __init__(self, name: str, providers: Sequence[Provider]):
        if not providers:
            raise ValueError(f"{name}: no providers")
        self.name = name
        self.providers = list(providers)

    async def call(self, *args, **kwargs) -> CallResult:
        now = time.time()
        downgrades: list[str] = []
        last_exc: BaseException | None = None
        for p in self.providers:
            if not p.available(now):
                downgrades.append(f"{p.name}: cooling "
                                  f"({int(p.cooldown_s - (now - p.last_failure))}s)")
                continue
            p.calls += 1
            try:
                value = await p.fn(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                               # noqa: BLE001
                p.record_failure(exc, now)
                last_exc = exc
                downgrades.append(f"{p.name}: {p.last_error}")
                logger.warning(f"[{self.name}] {p.name} failed: {p.last_error}")
                continue
            if not value:
                # An empty result is a failure too: "Whisper empty" has hidden
                # a NameError before (18.11) and must not be mistaken for OK.
                err = RuntimeError("empty result")
                p.record_failure(err, now)
                last_exc = err
                downgrades.append(f"{p.name}: empty result")
                logger.warning(f"[{self.name}] {p.name} returned nothing")
                continue
            if p.failures:
                logger.info(f"[{self.name}] {p.name} recovered")
            p.record_success()
            if downgrades:
                logger.info(f"[{self.name}] served by {p.name} after "
                            f"{'; '.join(downgrades)}")
            return CallResult(value, p.name, downgrades)
        raise RuntimeError(f"{self.name}: all providers failed"
                           + (f" (last: {last_exc!r})" if last_exc else ""))

    def status(self) -> dict:
        now = time.time()
        return {p.name: {"calls": p.calls, "failures": p.failures,
                         "available": p.available(now), "last": p.last_error}
                for p in self.providers}
