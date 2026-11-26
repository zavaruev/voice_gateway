"""Thin identity-service client used by the gateway.

Every call is optional: when IDENTITY_URL is empty the module-level IDENTITY
stays None and all existing call sites keep their Phase A behaviour exactly.
When identity fails the caller gets IdentityError and decides how to fall back.
"""
import asyncio
import os
import time

import aiohttp


class IdentityError(Exception):
    def __init__(self, status: int, detail: str = ""):
        self.status = int(status)
        self.detail = str(detail)[:300]
        super().__init__(f"identity HTTP {self.status}: {self.detail}")


class IdentityClient:
    def __init__(self, url: str, token: str, ttl: float = 30.0,
                 session=None, clock=None):
        self.url = str(url).rstrip("/")
        self.token = str(token or "")
        self.ttl = float(ttl)
        self.http = session
        self.clock = clock or time.monotonic
        self._cache = {}
        self._cached_at = None
        self._lock = None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    async def _request(self, method: str, path: str, form=None) -> dict:
        owner = self.http is None
        sess = self.http or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10))
        try:
            kwargs = {"headers": self._headers()}
            if form is not None:
                kwargs["data"] = form
            call = sess.post(f"{self.url}{path}", **kwargs) if method == "POST" \
                else sess.get(f"{self.url}{path}", **kwargs)
            async with call as r:
                if r.status != 200:
                    raise IdentityError(r.status, await r.text())
                return await r.json()
        finally:
            if owner:
                await sess.close()

    async def profiles(self) -> dict:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = self.clock()
            if self._cached_at is not None and now - self._cached_at < self.ttl:
                return self._cache
            data = await self._request("GET", "/api/profiles")
            self._cache = {p["key"]: p for p in data.get("profiles", [])}
            self._cached_at = self.clock()
            return self._cache

    async def identify(self, audio: bytes) -> tuple:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="audio.ogg", content_type="audio/ogg")
        data = await self._request("POST", "/api/identify", form)
        return (str(data.get("person", "unknown")),
                str(data.get("honorific", "") or ""),
                float(data.get("confidence", 0.0)))

    async def recognize(self, image: bytes, threshold: float = 0.35) -> list:
        form = aiohttp.FormData()
        form.add_field("file", image, filename="frame.jpg", content_type="image/jpeg")
        form.add_field("threshold", str(threshold))
        data = await self._request("POST", "/api/recognize", form)
        return data.get("faces", []) or []

    def chat_map(self) -> dict:
        return {
            str(chat): person
            for person, profile in self._cache.items()
            for chat in (profile.get("telegram_chat_ids") or [])
        }

    def honorific_for(self, person: str) -> str:
        return str((self._cache.get(str(person).lower()) or {}).get("honorific", "") or "")


IDENTITY: IdentityClient | None = None


def init_identity() -> None:
    global IDENTITY
    url = os.getenv("IDENTITY_URL", "").strip()
    IDENTITY = None if not url else IdentityClient(
        url, os.getenv("IDENTITY_TOKEN", ""))
