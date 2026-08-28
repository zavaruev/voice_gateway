import asyncio
import json
import re
import logging
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger("backends")

class BaseLLMBackend(ABC):
    @abstractmethod
    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        """
        Generates a response for the given text and pushes sentences into response_queue.
        Should put None into response_queue when finished.
        """
        pass

class NanobotBackend(BaseLLMBackend):
    def __init__(self, url: str, token: str, salt: str):
        self.url = url.rstrip("/")
        self.token = token
        self.salt = salt

    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"{self.url}?token={self.token}&chat_id={session_id}"
                ) as ws:
                    await ws.send_json({
                        "type": "message",
                        "chat_id": session_id,
                        "content": text,
                        "user_id": "camera",
                        "user_name": stream_name,
                        "voice_reply": True,
                    })
                    
                    _n_sent = 0
                    buf = ""
                    spoke_any = False

                    def speakable(s: str) -> str:
                        s = re.sub(r"\[[a-z]{2,30}\]", "", s)
                        return re.sub(r"\s+", " ", s).strip()

                    def flush(final: bool) -> None:
                        nonlocal buf, spoke_any, _n_sent
                        m = re.search(r"\[thinking\]", buf)
                        if m and not re.search(r"\[/thinking\]", buf[m.start():]):
                            part, holder = (buf[:m.start()], buf[m.start:]) if not final else (buf, "")
                        else:
                            idx = buf.rfind("[")
                            if idx != -1 and not re.search(r"\]", buf[idx:]):
                                part, holder = buf[:idx], buf[idx:]
                            else:
                                part, holder = buf, ""
                        
                        while True:
                            mm = re.search(r"\[thinking\](.*?)\[/thinking\]", part, flags=re.S)
                            if not mm: break
                            part = part[:mm.start()] + part[mm.end():]
                        
                        mm = re.search(r"^(.*[.!?…])([^.!?…]*)$", part, flags=re.S)
                        if mm:
                            done = [mm.group(1)] if mm.group(1).strip() else []
                            tail = mm.group(2)
                        else:
                            done, tail = [], part
                        
                        if final and tail.strip():
                            done.append(tail)
                            tail = ""
                        
                        for s in done:
                            s2 = speakable(s)
                            if s2:
                                spoke_any = True
                                _n_sent += 1
                                response_queue.put_nowait(s2)
                        buf = (tail + " " + holder).strip()

                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            ev = data.get("event")
                            if ev in ("done", "error", "final"):
                                break
                            if ev == "stream_end":
                                break
                            delta = None
                            if "text" in data and data.get("type") not in ("stt", "listen") and ev not in ("reasoning_delta", "thinking", "ready"):
                                delta = str(data["text"])
                            elif "parts" in data:
                                delta = str(data["parts"])
                            if delta:
                                buf += delta
                                if re.search(r"[.!?…](\s|$)", buf) or "\n" in buf or len(buf) > 400:
                                    flush(final=False)
                    finally:
                        flush(final=True)
                        await response_queue.put(None)
        except Exception as e:
            logger.error(f"Nanobot error: {e}")
            await response_queue.put(None)

class HermesBackend(BaseLLMBackend):
    def __init__(self, url: str, api_key: str = ""):
        self.url = url.rstrip("/")
        self.api_key = api_key

    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        try:
            async with aiohttp.ClientSession() as session:
                # Using the API server endpoint. 
                # If streaming is not available, we'll fetch the whole response and split by sentences.
                payload = {
                    "message": text,
                    "session_id": session_id,
                    "user_name": stream_name
                }
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                
                # Hermes API Server: OpenAI-compatible /v1/chat/completions
                payload = {
                    "model": "omniroute/oc/hy3-free",
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                }
                async with session.post(f"{self.url}/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            reply = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError):
                            reply = ""
                        if reply:
                            # Split into sentences to maintain the flow for TTS
                            sentences = re.split(r"(?<=[.!?…])\s+", reply)
                            for s in sentences:
                                if s.strip():
                                    response_queue.put_nowait(s.strip())
                        await response_queue.put(None)
                    else:
                        logger.error(f"Hermes API error: {resp.status}")
                        await response_queue.put(None)
        except Exception as e:
            logger.error(f"Hermes error: {e}")
            await response_queue.put(None)
