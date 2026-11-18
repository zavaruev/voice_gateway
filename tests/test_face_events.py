"""Tests for `face_events.py` — the stage-2 watcher ported out of the
face-greeting-handler container.

The most important assertion is the LAST one: the gateway must speak through
face_greeting + voice_mcp, never re-open its own /play_audio path. A second
voice stack is the bug this whole migration exists to remove.
"""

import asyncio
import json
import os
import time

import pytest

import face_events as fe


def _recognized(name, conf, frame="f.jpg", cam="camera_entrance1"):
    """The RECOGNIZED shape: (name, confidence, frame_path, camera_id)."""
    return (name, conf, frame, cam)


def _vote(name, conf, frame="f.jpg", cam="camera_entrance1"):
    # Vote entries are (confidence, frame_path, camera_id) — NAME is the dict
    # key, not part of the entry (the 4-tuple is the RECOGNIZED output shape).
    return (conf, frame, cam)


# --- the vote rule (ported verbatim from the handler) ----------------------


def test_two_frames_are_required():
    votes = {"alexander": [_vote("alexander", 0.45)]}
    assert fe.votes_to_recognized(votes, "12-00-00") == []
    votes["alexander"].append(_vote("alexander", 0.41, "f2.jpg"))
    out = fe.votes_to_recognized(votes, "12-00-00")
    assert [r[0] for r in out] == ["alexander"]


def test_best_sample_wins_and_a_close_second_identity_is_dropped():
    votes = {
        "alexander": [_vote("alexander", 0.45), _vote("alexander", 0.61, "b.jpg")],
        "vladislava": [_vote("vladislava", 0.58), _vote("vladislava", 0.59)],
    }
    out = fe.votes_to_recognized(votes, "12-00-00")
    assert len(out) == 1 and out[0][0] == "alexander"
    assert out[0][1] == 0.61                       # max over the samples
    assert out[0][2] == "b.jpg"


def test_a_clearly_separated_second_person_is_kept():
    votes = {
        "alexander": [_vote("alexander", 0.9), _vote("alexander", 0.8)],
        "marina": [_vote("marina", 0.5), _vote("marina", 0.5)],
    }
    out = fe.votes_to_recognized(votes, "12-00-00")
    assert {r[0] for r in out} == {"alexander", "marina"}


def test_area_weighting_can_override_a_single_distant_high_score():
    votes = {
        "alexander": [(0.50, "small.jpg", "c1"), (0.50, "big.jpg", "c2")],
        "vladislava": [(0.60, "far1.jpg", "c1"), (0.40, "far2.jpg", "c2")],
    }
    areas = {"small.jpg": 60, "big.jpg": 600,
             "far1.jpg": 60, "far2.jpg": 60}
    raw = fe.votes_to_recognized(votes, "12-00-00")
    assert raw[0][0] == "vladislava"          # the old max-confidence rule
    weighted = fe.votes_to_recognized(votes, "12-00-00", areas)
    assert weighted[0][0] == "alexander"      # a large clear face wins
    assert len(weighted) == 1                 # the close second is still dropped


def test_area_weighting_needs_no_measurement_when_disabled():
    votes = {"alexander": [(0.42, "a.jpg", "c1"), (0.41, "b.jpg", "c2")]}
    assert fe.votes_to_recognized(votes, "x")[0][1] == 0.42


def test_face_areas_come_from_the_daemons_meta(tmp_path):
    (tmp_path / "12-00-00_meta.json").write_text(__import__("json").dumps({
        "frame_details": [
            {"path": "/a.jpg", "face_size": 300, "quality_ok": True},
            {"path": "/b.jpg", "face_size": 80, "quality_ok": False},
        ]}))
    assert fe.face_areas_from_meta(str(tmp_path), "12-00-00") == {"/a.jpg": 300}


def test_marina_alone_is_not_greeted_but_with_company_she_is():
    assert fe.who_to_greet([_recognized("marina", 0.4)]) == set()
    assert fe.who_to_greet([_recognized("marina", 0.4),
                            _recognized("alexander", 0.9)]) == {"marina", "alexander"}


# --- event discovery / dedupe ---------------------------------------------


def test_find_new_events_distinguishes_faces_from_unknown(tmp_path, monkeypatch):
    day = tmp_path / "2026-11-18"
    day.mkdir()
    (day / "10-00-00_camera_entrance1_face01.jpg").write_bytes(b"x")
    (day / "10-01-00_unknown_meta.json").write_text("{}")
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    faces, unknown = fe.find_new_events(str(day), set())
    assert faces == {"10-00-00"}
    assert unknown == {"10-01-00"}
    processed = {fe.event_key(str(day), "10-00-00")}
    faces, unknown = fe.find_new_events(str(day), processed)
    assert faces == set()


def test_existing_events_marks_only_OLD_events_as_processed(tmp_path, monkeypatch):
    day = tmp_path / "2026-11-18"
    day.mkdir()
    old = day / "09-00-00_camera_entrance1_face01.jpg"
    old.write_bytes(b"x")
    fresh = day / "10-00-00_camera_entrance1_face01.jpg"
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    done = fe.existing_events()
    assert fe.event_key(str(day), "09-00-00") in done
    assert fe.event_key(str(day), "10-00-00") not in done


def test_an_event_without_meta_is_still_preloaded(tmp_path, monkeypatch):
    day = tmp_path / "2026-11-17"
    day.mkdir()
    f = day / "08-00-00_camera_entrance2_unknown.jpg"
    f.write_bytes(b"x")
    os.utime(f, (time.time() - 3600, time.time() - 3600))
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    assert fe.event_key(str(day), "08-00-00") in fe.existing_events()


def test_event_frames_are_ranked_by_the_daemons_verdicts(tmp_path):
    good = tmp_path / "b.jpg"
    bad = tmp_path / "a.jpg"
    good.write_bytes(b"x")
    bad.write_bytes(b"x")
    (tmp_path / "10-00-00_meta.json").write_text(json.dumps({"frame_details": [
        {"path": str(bad), "quality_ok": False, "face_size": 500, "det_score": 0.9},
        {"path": str(good), "quality_ok": True, "face_size": 100, "det_score": 0.7},
    ]}))
    assert fe.event_frames(str(tmp_path), "10-00-00")[0] == str(good)


def test_frame_verdicts_wait_for_the_daemon_to_publish(tmp_path):
    (tmp_path / "10-00-00_frames.jsonl").write_text(
        json.dumps({"path": "/x/a.jpg", "quality_ok": True}) + "\n" + "garbage\n")
    assert fe.frame_verdicts(str(tmp_path), "10-00-00") == {"/x/a.jpg": True}


# --- one voice, not two ----------------------------------------------------


def test_greet_uses_face_greeting_and_voice_mcp_not_a_camera_post(monkeypatch):
    """A regression to the old edge-tts/direct-/play_audio stack must fail."""
    watcher = fe.FaceGreetingWatcher()
    monkeypatch.setattr(fe, "TG_TOKEN", "")
    monkeypatch.setattr(fe, "FACE_ROOM", "entrance1")
    calls = {}

    def fake_prepare(names):
        calls["names"] = names
        return ("Добрый вечер, Александр.", "llm", "evening")

    async def fake_speak(text, room, bypass_quiet=False):
        calls.update(text=text, room=room, bypass=bypass_quiet)
        return '{"spoken": true, "room": "entrance1"}'

    monkeypatch.setattr(fe.face_greeting, "prepare", fake_prepare)
    monkeypatch.setattr(fe.voice_mcp, "_speak_tool", fake_speak)

    def forbidden_session(*a, **kw):
        raise AssertionError("a second HTTP/TTS stack was opened")

    monkeypatch.setattr(fe.aiohttp, "ClientSession", forbidden_session)
    asyncio.run(watcher._greet_and_notify([_recognized("alexander", 0.5)], "12-00-00"))
    assert calls["names"] == ["alexander"]
    assert calls["text"] == "Добрый вечер, Александр."
    assert calls["room"] == "entrance1"
    assert calls["bypass"] is True   # a real arrival at night is not proactive


def test_a_refused_speak_is_not_reported_as_a_greeting_in_telegram(monkeypatch):
    watcher = fe.FaceGreetingWatcher()
    monkeypatch.setattr(fe, "TG_TOKEN", "")
    monkeypatch.setattr(fe.face_greeting, "prepare",
                        lambda names: ("тест", "llm", "evening"))
    monkeypatch.setattr(fe.voice_mcp, "_speak_tool",
                        lambda *a, **kw: _json_err())
    captions = []

    async def fake_tg(recognized, prefix, greeting):
        captions.append(greeting)

    monkeypatch.setattr(watcher, "_telegram_notification", fake_tg)
    asyncio.run(watcher._greet_and_notify([_recognized("alexander", 0.5)], "12-00-00"))
    assert captions == [""]


async def _json_err():
    return '{"error": "room entrance1 spoke 30s ago (min gap 120s)"}'


def test_watcher_sees_a_new_event_and_speaks_exactly_once(tmp_path, monkeypatch):
    """End-to-end of the watcher loop on fake files: discover -> vote -> speak
    -> mark processed. No camera, no Telegram, no filesystem side effects.
    """
    day = tmp_path / "2026-11-18"
    day.mkdir()
    frame = day / "12-00-00_camera_entrance1_face01.jpg"
    frame.write_bytes(b"x")
    frame2 = day / "12-00-00_camera_entrance2_face01.jpg"
    frame2.write_bytes(b"x")
    (day / "12-00-00_frames.jsonl").write_text(
        json.dumps({"path": str(frame), "quality_ok": True}) + "\n" +
        json.dumps({"path": str(frame2), "quality_ok": True}) + "\n")
    (day / "12-00-00_meta.json").write_text("{}")
    monkeypatch.setattr(fe, "CAPTURES_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "PENDING_DIR", str(tmp_path / "pending"))
    monkeypatch.setattr(fe, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(fe, "STREAM_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(fe, "META_WAIT_SECONDS", 2)
    monkeypatch.setattr(fe, "TG_TOKEN", "")

    class _Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(fe.aiohttp, "ClientSession", lambda *a, **k: _Sess())

    async def fake_recognize(self, session, path, threshold):
        return [{"name": "alexander", "confidence": 0.5}]

    spoken = []

    async def fake_speak(text, room, bypass_quiet=False):
        spoken.append((text, room, bypass_quiet))
        return '{"spoken": true}'

    monkeypatch.setattr(fe.FaceGreetingWatcher, "_recognize_frame", fake_recognize)
    monkeypatch.setattr(fe.face_greeting, "prepare",
                        lambda names: ("Привет", "llm", "day"))
    monkeypatch.setattr(fe.voice_mcp, "_speak_tool", fake_speak)

    async def run():
        watcher = fe.FaceGreetingWatcher()
        task = asyncio.create_task(watcher.run())
        for _ in range(100):
            if watcher.processed_events:
                break
            await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return watcher

    watcher = asyncio.run(run())
    assert fe.event_key(str(day), "12-00-00") in watcher.processed_events
    assert len(spoken) == 1, spoken
    assert spoken[0][1] == "entrance1" and spoken[0][2] is True
