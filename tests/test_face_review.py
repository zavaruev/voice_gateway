"""Tests for the face review loop (Telegram inline buttons -> pending/_review).

callback_data comes from a chat, so the name in it is never trusted as a path:
it must be one of the names offered in the stored record.
"""

import json

import pytest

import face_review as fr


class _Form:
    def __init__(self):
        self.fields = {}

    def add_field(self, key, value, **kw):
        self.fields[key] = value


def test_callback_roundtrip_and_rejection():
    cb = fr.make_callback("a1b2c3d4e5f6", "alexander")
    assert fr.parse_callback(cb) == ("a1b2c3d4e5f6", "alexander")
    assert fr.parse_callback("rv:short:alexander") is None
    assert fr.parse_callback("rv:a1b2c3d4e5f6:../etc/passwd") is None
    assert fr.parse_callback("garbage") is None


def test_apply_choice_copies_the_frame_only_for_an_offered_name(tmp_path):
    frame = tmp_path / "face.jpg"
    frame.write_bytes(b"jpeg")
    review = tmp_path / "records"
    pending = tmp_path / "pending"
    fr.save_record(str(review), "a1b2c3d4e5f6", str(frame),
                   ["alexander", "unknown"], "2026-11-18/12-00-00")
    assert "ignored" in fr.apply_choice(
        str(review), str(pending), "rv:a1b2c3d4e5f6:vladislava")
    assert not (pending / "_review" / "vladislava").exists()
    verdict = fr.apply_choice(str(review), str(pending),
                              "rv:a1b2c3d4e5f6:alexander")
    assert verdict.startswith("reviewed as alexander")
    assert (pending / "_review" / "alexander" / "face.jpg").read_bytes() == b"jpeg"
    # The record is consumed: replaying the same callback cannot duplicate it.
    assert "expired" in fr.apply_choice(str(review), str(pending),
                                        "rv:a1b2c3d4e5f6:alexander")


def test_unknown_is_not_a_negative_training_label(tmp_path):
    frame = tmp_path / "face.jpg"
    frame.write_bytes(b"jpeg")
    review = tmp_path / "records"
    pending = tmp_path / "pending"
    fr.save_record(str(review), "b1b2c3d4e5f6", str(frame), ["unknown"], "x")
    fr.apply_choice(str(review), str(pending), "rv:b1b2c3d4e5f6:unknown")
    assert not (pending / "_review" / "unknown").exists()


@pytest.mark.asyncio
async def test_send_photo_includes_the_inline_keyboard(monkeypatch, tmp_path):
    import face_events as fe

    fe.TG_TOKEN = "test-token"
    fe.FACE_REVIEW_ENABLED = True
    fe.REVIEW_DIR = str(tmp_path / "records")
    fe.KNOWN_NAMES = {"alexander"}
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    captured = {}

    class _Resp:
        status = 200

        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def text(self): return ""

    class _Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        def post(self, url, **kw):
            captured.update(kw)
            return _Resp()

    monkeypatch.setattr(fe.aiohttp, "ClientSession", lambda *a, **k: _Sess())
    monkeypatch.setattr(fe.aiohttp, "FormData", _Form)
    watcher = fe.FaceGreetingWatcher()
    await watcher._send_photo(str(frame), "caption", "alexander",
                              choices=["alexander", "unknown"])
    markup = json.loads(captured["data"].fields["reply_markup"])
    buttons = [b["callback_data"] for row in markup["inline_keyboard"] for b in row]
    assert len(buttons) == 2
    assert all(fr.parse_callback(b) for b in buttons)
