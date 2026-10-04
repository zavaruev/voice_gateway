"""Tests for telegram_client.py — the Telegram source (no network, no bot lib).

Pattern: fake TelegramAPI + scripted fake backend, like tests/test_cascade_
backend.py but with the api object injected instead of a fake aiohttp stack
(the bot calls TelegramAPI, not aiohttp, directly). Everything is driven
through asyncio.run() — no pytest-asyncio dependency, same as the backend
tests.

The heavy audio helpers are monkeypatched at module level (the call sites
reference the module globals), so no ffmpeg/pydub runs in these tests.
"""

import asyncio
import json
import os
import tempfile

import pytest

import telegram_client as tg
from telegram_client import (
    TelegramBot,
    is_valid_voice_text,
    normalize_room,
    split_reply,
)


# --------------------------------------------------------------------------- fakes


class FakeAPI:
    """Records every Bot API call the bot makes."""

    def __init__(self):
        self.messages: list[dict] = []
        self.edits: list[dict] = []
        self.actions: list[str] = []
        self.voices: list[dict] = []
        self.files: dict[str, tuple[str, bytes]] = {}
        self._mid = 100

    async def get_me(self):
        return {"username": "test_bot"}

    async def delete_webhook(self, drop_pending_updates=False):
        return True

    async def get_updates(self, offset, timeout=25):
        return []

    async def send_message(self, chat_id, text, reply_to=None):
        self._mid += 1
        self.messages.append(
            {"chat_id": chat_id, "text": text, "reply_to": reply_to, "id": self._mid}
        )
        return self._mid

    async def edit_message(self, chat_id, message_id, text):
        self.edits.append({"chat_id": chat_id, "id": message_id, "text": text})
        return True

    async def send_chat_action(self, chat_id):
        self.actions.append(chat_id)
        return True

    async def get_file(self, file_id):
        return self.files[file_id][0]

    async def download_file(self, file_path):
        for _fid, (path, data) in self.files.items():
            if path == file_path:
                return data
        raise FileNotFoundError(file_path)

    async def send_voice(self, chat_id, ogg, reply_to=None):
        self.voices.append({"chat_id": chat_id, "ogg": ogg, "reply_to": reply_to})
        return 1

    # helpers for assertions -------------------------------------------------
    def texts(self):
        return [m["text"] for m in self.messages]

    def edited_texts(self):
        return [e["text"] for e in self.edits]


class ScriptedBackend:
    """generate_response pushes the scripted items, then the None sentinel."""

    def __init__(self, items):
        self.items = items
        self.calls: list[dict] = []

    async def generate_response(self, text, session_id, stream_name,
                                response_queue, room=""):
        self.calls.append(
            {
                "text": text,
                "session_id": session_id,
                "stream_name": stream_name,
                "room": room,
            }
        )
        for item in self.items:
            response_queue.put_nowait(item)
        response_queue.put_nowait(None)


class HangingBackend:
    """Sends one sentence and then never terminates (for the timeout test)."""

    def __init__(self):
        self.calls = []

    async def generate_response(self, text, session_id, stream_name,
                                response_queue, room=""):
        self.calls.append({"text": text, "room": room})
        response_queue.put_nowait("Начало ответа.")
        await asyncio.sleep(3600)


class SilentHangingBackend:
    """Sends NOTHING and hangs — the pure fallback-timeout case."""

    def __init__(self):
        self.calls = []

    async def generate_response(self, text, session_id, stream_name,
                                response_queue, room=""):
        self.calls.append({"text": text, "room": room})
        await asyncio.sleep(3600)


async def _fake_transcribe(ogg, sess):
    return "включи свет"


async def _no_tts(text, sess):
    # Default: TTS "unavailable" — keeps ffmpeg out of tests that do not
    # care about the voice reply.
    return None


async def _mp3_tts(text, sess):
    return b"MP3"


async def _hallucination_tts(ogg, sess):
    return "спасибо за просмотр"


def _make_bot(tmp_path=None, *, allowed=("111",), backend=None, **kw):
    api = FakeAPI()
    state_file = os.path.join(tmp_path or tempfile.mkdtemp(), "tg.json")
    bot = TelegramBot(
        api=api,
        allowed_chat_ids=allowed,
        backend=backend or ScriptedBackend(["Привет.", "Как дела?"]),
        transcribe=kw.pop("transcribe", None) or _fake_transcribe,
        tts_mp3=kw.pop("tts_mp3", None) or _no_tts,
        make_session_id=lambda chat: f"sess-{chat}",
        state_file=state_file,
        # Throttle off by default: tests must not sleep between turns.
        cooldown=kw.pop("cooldown", 0.0),
        **kw,
    )
    return bot, api


def _msg(chat_id=111, message_id=7, **extra):
    base = {"chat": {"id": chat_id, "type": "private"}, "message_id": message_id}
    base.update(extra)
    return base


def _update(chat_id=111, **extra):
    """A Bot API update envelope around a message (what dispatch() expects)."""
    return {"update_id": 1, "message": _msg(chat_id=chat_id, **extra)}


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------- pure helpers


def test_normalize_room_accepts_ru_and_en():
    assert normalize_room("кухня") == "kitchen"
    assert normalize_room("КУХНЕ") == "kitchen"
    assert normalize_room("на кухне") == "kitchen"
    assert normalize_room("living room") in ("livingroom", "living_room")
    assert normalize_room("corridor") == "corridor"
    # "hallway" is a STREAM_DEFAULT_AREA key in its own right (-> Corridor).
    assert normalize_room("hallway") in ("corridor", "hallway")
    assert normalize_room("спальня") == "bedroom"
    assert normalize_room("в ванной") is None
    assert normalize_room("") is None


def test_valid_voice_text_accepts_real_short_answers():
    # The whole point of the Telegram variant: «да» (2 chars) must pass —
    # the device filter rejects it and would break the answer-to-a-question
    # flow.
    assert is_valid_voice_text("да")
    assert is_valid_voice_text("ок")
    assert is_valid_voice_text("ага, включи")
    assert is_valid_voice_text("Включи свет на кухне.")


def test_valid_voice_text_rejects_whisper_garbage():
    assert not is_valid_voice_text("")
    assert not is_valid_voice_text("   ")
    assert not is_valid_voice_text("о")           # silence hallucination
    assert not is_valid_voice_text("как")         # in SINGLE_WORD_HALLUCINATIONS
    assert not is_valid_voice_text("ккккккккк")   # char run
    assert not is_valid_voice_text("каккаккак")   # repeated substring
    assert not is_valid_voice_text("да да да да")  # model looping
    assert not is_valid_voice_text("спасибо за просмотр")  # classic hallucination


def test_split_reply_short_text_is_one_chunk():
    assert split_reply("Короткий ответ.") == ["Короткий ответ."]
    assert split_reply("") == []


def test_split_reply_long_text_respects_limit():
    text = " ".join(f"Предложение номер {i} идёт дальше." for i in range(200))
    chunks = split_reply(text, limit=400)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # No content lost or duplicated.
    assert " ".join(chunks) == text
    # Splits land on sentence ends, not mid-word.
    assert all(c.rstrip().endswith(".") for c in chunks[:-1])


# ------------------------------------------------------------------ access


def test_allowlist_blocks_unknown_chat():
    bot, api = _make_bot(allowed=("111",))
    _run(bot.handle_message(222, _msg(chat_id=222, text="привет")))
    assert api.messages == []


def test_empty_allowlist_denies_everyone():
    bot, api = _make_bot(allowed=())
    _run(bot.handle_message(111, _msg(text="привет")))
    assert api.messages == []


def test_dispatch_gate_never_queues_rejected_chats():
    bot, _api = _make_bot(allowed=("111",))
    bot.dispatch(_update(chat_id=222, text="hack"))
    assert bot._queues == {}  # no worker even created


def test_group_messages_ignored_by_default():
    bot, _api = _make_bot(allowed=("-100",))
    bot.dispatch({"update_id": 1, "message": {
        "chat": {"id": -100, "type": "group"}, "message_id": 1, "text": "привет"}})
    assert bot._queues == {}


def test_groups_allowed_with_flag():
    async def flow():
        bot, _api = _make_bot(allowed=("-100",), allow_groups=True)
        bot.dispatch({"update_id": 1, "message": {
            "chat": {"id": -100, "type": "group"}, "message_id": 1, "text": "привет"}})
        assert "-100" in bot._queues
        await bot._stop_workers()

    _run(flow())


# -------------------------------------------------------------- text turns


def test_text_turn_streams_reply_and_uses_unique_stream_name():
    backend = ScriptedBackend(["Отвечаю."])
    bot, api = _make_bot(backend=backend)
    _run(bot.handle_message(111, _msg(text="привет")))

    call = backend.calls[0]
    assert call["text"] == "привет"
    assert call["session_id"] == "sess-111"
    assert call["stream_name"] == "tg:111"   # unique per chat, never a room
    assert call["room"] == ""
    assert any("Отвечаю." in t for t in api.texts() + api.edited_texts())


def test_different_chats_get_different_history_keys():
    backend = ScriptedBackend(["ок"])
    bot, _api = _make_bot(backend=backend, allowed=("111", "222"))
    _run(bot.handle_message(111, _msg(chat_id=111, text="один")))
    _run(bot.handle_message(222, _msg(chat_id=222, text="два")))
    keys = {c["stream_name"] for c in backend.calls}
    assert keys == {"tg:111", "tg:222"}


def test_ack_phrases_become_status_and_never_enter_final_text():
    from backends import CascadeBackend
    backend = ScriptedBackend(
        [CascadeBackend.ACK_COMPLEX, "Проверил: свет выключен."]
    )
    bot, api = _make_bot(backend=backend)
    _run(bot.handle_message(111, _msg(text="выключи свет")))

    # Ack is its own bubble...
    assert any(t == CascadeBackend.ACK_COMPLEX for t in api.texts())
    # ...and the final reply (last edit or last message) must not contain it.
    final = api.edited_texts()[-1] if api.edited_texts() else api.texts()[-1]
    assert CascadeBackend.ACK_COMPLEX not in final
    assert "Проверил: свет выключен." in final


def test_timeout_sends_fallback_instead_of_hanging():
    bot, api = _make_bot(backend=SilentHangingBackend(), turn_timeout=0.2)
    out = _run(bot.handle_message(111, _msg(text="сложный вопрос")))
    assert out == ""
    assert any(tg.FALLBACK_TIMEOUT in t for t in api.texts())


def test_timeout_with_partial_sends_the_partial():
    bot, api = _make_bot(backend=HangingBackend(), turn_timeout=0.3)
    # HangingBackend's first sentence flushes immediately, so the partial
    # text is what the user gets (no fallback on top).
    out = _run(bot.handle_message(111, _msg(text="сложный вопрос")))
    assert out == "Начало ответа."
    joined = api.texts() + api.edited_texts()
    assert not any(tg.FALLBACK_TIMEOUT in t for t in joined)


def test_silent_backend_gets_an_apology():
    bot, api = _make_bot(backend=ScriptedBackend([None]))
    _run(bot.handle_message(111, _msg(text="привет")))
    assert api.messages or api.edits
    joined = api.texts() + api.edited_texts()
    assert any(tg.FALLBACK_ERROR in t for t in joined)


def test_backend_error_closing_phrase_is_not_doubled():
    from backends import CascadeBackend
    bot, api = _make_bot(backend=ScriptedBackend([CascadeBackend.SORRY, None]))
    _run(bot.handle_message(111, _msg(text="привет")))
    # SORRY and FALLBACK_ERROR are the same wording on purpose; doubling
    # would show up as two identical closers.
    assert api.texts().count(tg.FALLBACK_ERROR) == 1


def test_long_reply_overflows_into_extra_messages():
    long_text = " ".join(f"Это довольно длинное предложение номер {i}." for i in range(300))
    bot, api = _make_bot(backend=ScriptedBackend([long_text]))
    _run(bot.handle_message(111, _msg(text="расскажи подробно")))

    # The overflow tail reached the chat as extra message(s)...
    assert "предложение номер 299" in " ".join(api.texts())
    # ...nothing was ever edited over the message cap...
    assert all(len(e["text"]) <= tg.STREAM_LIMIT for e in api.edits)
    # ...and more than one message went out (stream bubble + overflow).
    assert len(api.messages) >= 2


# ------------------------------------------------------------- voice turns


def test_voice_turn_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tg, "prepare_stt_audio", lambda data, ext, dur: (b"OGG", dur or 1.0)
    )
    backend = ScriptedBackend(["Свет включён."])
    bot, api = _make_bot(tmp_path=str(tmp_path), backend=backend)
    api.files["f1"] = ("voice.ogg", b"RAWVOICE")
    msg = _msg(voice={"file_id": "f1", "duration": 3}, text=None)
    _run(bot.handle_message(111, msg))

    assert backend.calls[0]["text"] == "включи свет"
    assert any("Свет включён." in t for t in api.edited_texts() + api.texts())


def test_voice_too_long_refused_before_download(tmp_path):
    bot, api = _make_bot(tmp_path=str(tmp_path), max_voice_s=60)
    msg = _msg(voice={"file_id": "f1", "duration": 300})
    _run(bot.handle_message(111, msg))
    # The guard tripped before any download and before any LLM turn.
    assert any("Слишком длинное" in t for t in api.texts())
    assert bot.backend.calls == []


def test_voice_unreadable_audio_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "prepare_stt_audio", lambda data, ext, dur: (None, 0.0))
    bot, api = _make_bot(tmp_path=str(tmp_path))
    api.files["f1"] = ("voice.ogg", b"BROKEN")
    _run(bot.handle_message(111, _msg(voice={"file_id": "f1", "duration": 3})))
    assert any(tg.HINT_DOWNLOAD_FAILED in t for t in api.texts())


def test_voice_hallucination_transcript_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tg, "prepare_stt_audio", lambda data, ext, dur: (b"OGG", dur or 1.0)
    )
    bot, api = _make_bot(
        tmp_path=str(tmp_path), transcribe=_hallucination_tts
    )
    api.files["f1"] = ("voice.ogg", b"RAWVOICE")
    _run(bot.handle_message(111, _msg(voice={"file_id": "f1", "duration": 3})))
    assert any(tg.HINT_NOT_HEARD in t for t in api.texts())


def test_voice_reply_sent_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "mp3_to_voice_ogg", lambda mp3: b"OGGVOICE")
    bot, api = _make_bot(tmp_path=str(tmp_path), tts_mp3=_mp3_tts)
    _run(bot.handle_message(111, _msg(text="привет")))
    assert api.voices and api.voices[0]["ogg"] == b"OGGVOICE"


def test_voice_reply_disabled_by_command(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "mp3_to_voice_ogg", lambda mp3: b"OGGVOICE")
    bot, api = _make_bot(tmp_path=str(tmp_path), tts_mp3=_mp3_tts)

    async def flow():
        await bot.handle_message(111, _msg(text="/voice off"))
        await bot.handle_message(111, _msg(text="привет"))

    _run(flow())
    assert api.voices == []
    assert any("отвечаю текстом" in t for t in api.texts())


def test_voice_reply_respects_global_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "mp3_to_voice_ogg", lambda mp3: b"OGGVOICE")
    bot, api = _make_bot(tmp_path=str(tmp_path), reply_voice=False)
    _run(bot.handle_message(111, _msg(text="привет")))
    assert api.voices == []


# ---------------------------------------------------------------- commands


def test_room_command_binds_room_and_passes_it_to_backend(tmp_path):
    backend = ScriptedBackend(["Ок."])
    bot, api = _make_bot(tmp_path=str(tmp_path), backend=backend)

    async def flow():
        await bot.handle_message(111, _msg(text="/room кухня"))
        await bot.handle_message(111, _msg(text="включи свет"))

    _run(flow())
    assert bot.chat_room("111") == "kitchen"
    assert backend.calls[0]["room"] == "kitchen"
    # And the binding survived to disk for the next container start.
    with open(bot.state_file) as f:
        assert json.load(f)["111"]["room"] == "kitchen"


def test_room_command_rejects_unknown_room(tmp_path):
    bot, api = _make_bot(tmp_path=str(tmp_path))
    _run(bot.handle_message(111, _msg(text="/room в ванной")))
    assert bot.chat_room("111") == ""
    assert any("Не знаю комнаты" in t for t in api.texts())


def test_room_shows_current_binding(tmp_path):
    bot, api = _make_bot(tmp_path=str(tmp_path))
    _run(bot.handle_message(111, _msg(text="/room")))
    assert any("не задана" in t for t in api.texts())


def test_state_survives_restart(tmp_path):
    async def flow():
        bot1, _ = _make_bot(tmp_path=str(tmp_path))
        await bot1.handle_message(111, _msg(text="/room коридор"))
        return bot1

    _run(flow())
    bot2, _api = _make_bot(tmp_path=str(tmp_path))
    assert bot2.chat_room("111") == "corridor"


def test_start_command_greets():
    bot, api = _make_bot()
    _run(bot.handle_message(111, _msg(text="/start")))
    assert any("голосовой ассистент" in t for t in api.texts())


def test_unsupported_content_gets_a_hint():
    bot, api = _make_bot()
    _run(bot.handle_message(111, _msg(sticker={"emoji": "👍"})))
    assert api.texts() == [tg.HINT_UNSUPPORTED]


# ----------------------------------------------------------------- flow


def test_dispatch_orders_updates_inside_one_chat():
    processed = []

    class Recorder(ScriptedBackend):
        async def generate_response(self, text, session_id, stream_name,
                                    response_queue, room=""):
            processed.append(text)
            await super().generate_response(
                text, session_id, stream_name, response_queue, room
            )

    bot, _api = _make_bot(backend=Recorder(["ок"]))

    async def flow():
        # Two updates from one chat must queue behind each other in order.
        bot.dispatch(_update(text="первое"))
        bot.dispatch(_update(text="второе"))
        for _ in range(300):
            if len(processed) >= 2:
                break
            await asyncio.sleep(0.01)
        await bot._stop_workers()

    _run(flow())
    assert processed == ["первое", "второе"]


def test_queue_cap_drops_flood():
    bot, _api = _make_bot(max_pending=2)

    async def flow():
        bot.dispatch(_update(text="1"))
        bot.dispatch(_update(text="2"))
        bot.dispatch(_update(text="3"))  # cap reached -> dropped
        size = bot._queues["111"].qsize()
        await bot._stop_workers()
        return size

    assert _run(flow()) == 2


def test_build_factory_wires_api_base():
    bot = tg.build_telegram_bot(
        token="T",
        allowed_chat_ids=["1"],
        backend=ScriptedBackend([]),
        transcribe=lambda a, s: "",
        tts_mp3=lambda t, s: None,
        make_session_id=lambda c: c,
        api_base="http://localhost:9999",
    )
    assert isinstance(bot, TelegramBot)
    assert bot.api.base == "http://localhost:9999"
    assert bot.api.token == "T"


def test_typing_action_runs_during_turn():
    backend = ScriptedBackend(["Ответ."])
    bot, api = _make_bot(backend=backend)
    _run(bot.handle_message(111, _msg(text="привет")))
    # At least one typing ping went out before the reply finished.
    assert "111" in api.actions


def test_send_failure_never_breaks_a_turn(tmp_path):
    """A dead Bot API must not raise out of a command/refusal path.

    The Bot API is the only channel we have — there is nothing to tell the
    user about a failed send, so it is logged and swallowed. (Regression:
    raw api.send_message() calls used to escape into the chat worker and
    look like a failed turn.)
    """
    bot, api = _make_bot(tmp_path=str(tmp_path))

    async def boom(*a, **k):
        raise RuntimeError("Bad Request: chat not found")

    api.send_message = boom

    async def flow():
        # Commands (2 sends), a too-long text (1 send) and an unsupported
        # type (1 send) must all return quietly...
        await bot.handle_message(111, _msg(text="/start"))
        await bot.handle_message(111, _msg(text="/room кухня"))
        await bot.handle_message(111, _msg(text="я" * 5000))
        await bot.handle_message(111, _msg(sticker={"emoji": "👍"}))
        # ...and the binding still got persisted despite the dead API.
        return bot.chat_room("111")

    assert _run(flow()) == "kitchen"


def test_send_failure_still_answers_the_llm_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "mp3_to_voice_ogg", lambda mp3: b"OGG")
    backend = ScriptedBackend(["Ответ."])
    bot, api = _make_bot(tmp_path=str(tmp_path), backend=backend,
                         tts_mp3=_mp3_tts)

    async def boom(*a, **k):
        raise RuntimeError("Bad Request: chat not found")

    api.send_message = boom
    out = _run(bot.handle_message(111, _msg(text="привет")))
    # The LLM ran and the text is still returned to the caller...
    assert out == "Ответ."
    assert backend.calls[0]["text"] == "привет"
