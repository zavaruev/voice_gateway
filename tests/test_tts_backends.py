"""Three switchable TTS backends (edge / coqui / silero), added 11.11.2026.

The user asked for a choice of engine at runtime after Coqui's voice quality
was rejected and playback stuttered. `TTS_BACKENDS` in main.py holds one preset
per engine; apply_tts_backend() rebinds the legacy TTS_* globals — which both
consumption paths read at call time — and updates live camera sessions in
place. Only coqui carries a stream_url (edge/silero are whole-file), which is
also the stutter-free path (chunked bodies are consumed slower by the cameras).

Zero regression rule: with TTS_BACKEND unset the legacy envs win EXACTLY as
before — nothing is applied at boot, so every default-path test stays green.
"""

import pytest

import main as main_mod

TTS_GLOBALS = ("TTS_URL", "TTS_MODEL", "TTS_VOICE", "TTS_STREAM_URL",
               "TTS_BACKEND_ACTIVE")


@pytest.fixture
def restore_tts():
    """Snapshot the TTS globals + the live-session list and restore them.

    apply_tts_backend() mutates module state shared with every other test;
    a test that leaks its switch would silently re-point someone else's TTS.
    """
    saved = {n: getattr(main_mod, n) for n in TTS_GLOBALS}
    saved_sessions = main_mod._camera_sessions
    yield
    for n, v in saved.items():
        setattr(main_mod, n, v)
    main_mod._camera_sessions = saved_sessions


class _StubSession:
    """The four session attributes the switcher writes (CameraSession sets
    all four from its CameraConfig in __init__)."""

    def __init__(self, name="livingroom"):
        self.stream_name = name
        self.tts_url = "http://old:1"
        self.tts_voice = "old-voice"
        self.tts_stream_url = "http://old:1/stream"


# --- presets -----------------------------------------------------------------


def test_three_presets_and_only_coqui_streams():
    """Only coqui carries a stream endpoint: edge and silero are whole-file
    services, and whole-file is the stutter-free path (11.11.2026: chunked
    bodies are consumed x2.7..x5.2 slower by the cameras)."""
    assert set(main_mod.TTS_BACKENDS) == {"edge", "coqui", "silero"}
    assert main_mod.TTS_BACKENDS["coqui"]["stream_url"]
    for name in ("edge", "silero"):
        assert main_mod.TTS_BACKENDS[name]["stream_url"] == "", (
            f"{name} must be a whole-file backend"
        )


def test_silero_preset_is_baya_v5_6_ru():
    p = main_mod.TTS_BACKENDS["silero"]
    assert p["voice"] == "baya"
    assert p["model"] == "v5_6_ru"
    assert p["url"] == "http://192.168.22.102:5053/v1/audio/speech"


def test_edge_preset_keeps_the_legacy_shape():
    p = main_mod.TTS_BACKENDS["edge"]
    assert p["voice"] == "ru-RU-SvetlanaNeural"
    assert p["model"] == "tts-1"
    assert p["url"] == "http://edge_tts:5050/v1/audio/speech"


# --- apply_tts_backend -------------------------------------------------------


def test_apply_rebinds_globals_and_live_sessions(restore_tts):
    stub = _StubSession()
    main_mod._camera_sessions = [stub]
    main_mod.TTS_BACKEND_ACTIVE = ""

    applied = main_mod.apply_tts_backend("silero")

    assert applied == "silero"
    assert main_mod.TTS_BACKEND_ACTIVE == "silero"
    p = main_mod.TTS_BACKENDS["silero"]
    assert main_mod.TTS_URL == p["url"]
    assert main_mod.TTS_MODEL == p["model"]
    assert main_mod.TTS_VOICE == "baya"
    assert main_mod.TTS_STREAM_URL == ""
    # The LIVE session follows the switch, not just future configs.
    assert stub.tts_url == p["url"]
    assert stub.tts_voice == "baya"
    assert stub.tts_stream_url == ""


def test_unknown_backend_changes_nothing(restore_tts):
    """A typo must not half-apply: KeyError with the globals untouched —
    the refusal direction of every guard in this repo."""
    saved = {n: getattr(main_mod, n) for n in TTS_GLOBALS}
    main_mod.TTS_BACKEND_ACTIVE = "coqui"
    saved["TTS_BACKEND_ACTIVE"] = "coqui"

    with pytest.raises(KeyError):
        main_mod.apply_tts_backend("sileroo")

    for n, v in saved.items():
        assert getattr(main_mod, n) == v, f"{n} moved on a refused switch"


def test_voice_override_wins_but_preset_stays(restore_tts):
    main_mod._camera_sessions = []
    main_mod.apply_tts_backend("silero", voice="kseniya")
    assert main_mod.TTS_VOICE == "kseniya"
    assert main_mod.TTS_BACKEND_ACTIVE == "silero"
    assert main_mod.TTS_URL == main_mod.TTS_BACKENDS["silero"]["url"]


def test_empty_voice_falls_back_to_preset(restore_tts):
    """`or` on purpose: an empty string from a JSON body must not wipe the
    voice to '' — the shape that once sent a dataclass default to a service
    that then answered in a man's voice (10.11.2026)."""
    main_mod._camera_sessions = []
    main_mod.apply_tts_backend("silero", voice="")
    assert main_mod.TTS_VOICE == "baya"


# --- boot application --------------------------------------------------------


def test_boot_backend_empty_keeps_legacy_envs(restore_tts, monkeypatch):
    """TTS_BACKEND unset => the legacy envs win EXACTLY as before. This is
    the zero-regression guarantee for every deployment that never switches."""
    monkeypatch.setattr(main_mod, "TTS_BACKEND", "")
    main_mod.TTS_URL = "http://legacy:1/v1/audio/speech"
    main_mod.TTS_VOICE = "legacy-voice"

    assert main_mod._boot_tts_backend() == ""
    assert main_mod.TTS_URL == "http://legacy:1/v1/audio/speech"
    assert main_mod.TTS_VOICE == "legacy-voice"
    assert main_mod.TTS_BACKEND_ACTIVE == ""


def test_boot_backend_applies_the_named_preset(restore_tts, monkeypatch):
    monkeypatch.setattr(main_mod, "TTS_BACKEND", "silero")
    assert main_mod._boot_tts_backend() == "silero"
    assert main_mod.TTS_VOICE == "baya"
    assert main_mod.TTS_BACKEND_ACTIVE == "silero"


# --- REST --------------------------------------------------------------------


@pytest.fixture
def api_client(restore_tts):
    from fastapi.testclient import TestClient

    main_mod.app.dependency_overrides[main_mod.verify_auth] = lambda: "admin"
    yield TestClient(main_mod.app)
    main_mod.app.dependency_overrides.clear()


def test_rest_lists_backends_and_active(api_client):
    r = api_client.get("/api/tts/backends")
    assert r.status_code == 200
    body = r.json()
    assert set(body["backends"]) == {"edge", "coqui", "silero"}
    assert "active" in body and "voice" in body


def test_rest_post_switches_and_reports(api_client):
    r = api_client.post("/api/tts/backends/silero", json={"voice": "xenia"})
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "silero"
    assert body["voice"] == "xenia"
    assert body["url"] == main_mod.TTS_BACKENDS["silero"]["url"]
    assert main_mod.TTS_VOICE == "xenia"


def test_rest_post_without_body_switches(api_client):
    """A bare POST (curl without -d) is valid: found live 11.11.2026 as a 500
    because req.json() raises on an empty body. json={} in a test sends a body
    and cannot see this — the field case must be reproduced exactly."""
    r = api_client.post("/api/tts/backends/silero")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "silero"
    assert body["voice"] == "baya", "a bare POST must take the preset voice"


def test_rest_post_malformed_body_is_400(api_client):
    r = api_client.post("/api/tts/backends/silero",
                        content=b"not json",
                        headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_rest_post_unknown_is_404_and_changes_nothing(api_client):
    main_mod.TTS_BACKEND_ACTIVE = "edge"
    main_mod.TTS_VOICE = "ru-RU-SvetlanaNeural"

    r = api_client.post("/api/tts/backends/nope", json={})

    assert r.status_code == 404
    assert "available" in r.json()["detail"]
    assert main_mod.TTS_BACKEND_ACTIVE == "edge", "a refused POST must not switch"
    assert main_mod.TTS_VOICE == "ru-RU-SvetlanaNeural"


# --- new camera sessions follow the active backend ---------------------------


@pytest.mark.asyncio
async def test_new_camera_session_follows_active_backend(restore_tts, monkeypatch):
    """start_camera_sessions() fills CameraConfig from the module globals, so a
    session started AFTER a switch must carry the switched URL/voice/stream —
    the same wiring the 10.11.2026 tts_voice fill bug lived in (a missing fill
    is invisible to any test of CameraConfig alone)."""
    main_mod.apply_tts_backend("silero")
    main_mod._camera_sessions = []

    created = []

    class _FakeSession:
        def __init__(self, config=None, backend=None):
            self.config = config
            created.append(self)

        async def start(self):
            return None

    class _StubBackend:
        pass

    class _FakeClientSession:
        def __init__(self, timeout=None):
            pass

    monkeypatch.delenv("DISABLE_CAMERAS", raising=False)
    monkeypatch.setenv("CAMERA_STREAMS", "livingroom")
    monkeypatch.setattr(main_mod, "CameraSession", _FakeSession)
    monkeypatch.setattr(main_mod, "_camera_sessions", [])
    monkeypatch.setattr(main_mod, "VadEngine", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "LLM_BACKEND", "cascade")
    monkeypatch.setattr(main_mod.backends, "CascadeBackend",
                        lambda *a, **k: _StubBackend())
    monkeypatch.setattr(main_mod.aiohttp, "ClientSession", _FakeClientSession)

    await main_mod.start_camera_sessions()

    assert len(created) == 1
    cfg = created[0].config
    p = main_mod.TTS_BACKENDS["silero"]
    assert cfg.tts_url == p["url"]
    assert cfg.tts_voice == "baya"
    # The whole-file path: a new session on a non-streaming backend must NOT
    # inherit a stream URL from a previous backend.
    assert cfg.tts_stream_url == ""
