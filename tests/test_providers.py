"""Tests for the provider chain: a fallback must be visible, not silent."""

import pytest

from providers import Provider, ProviderChain


@pytest.mark.asyncio
async def test_first_success_never_touches_the_fallback():
    calls = []

    async def good(v):
        calls.append("good")
        return "OK"

    async def backup(v):
        calls.append("backup")
        return "BACKUP"

    chain = ProviderChain("tts", [Provider("good", good), Provider("backup", backup)])
    res = await chain.call("x")
    assert res.value == "OK" and res.provider == "good"
    assert res.downgrades == []
    assert calls == ["good"]


@pytest.mark.asyncio
async def test_a_failure_falls_back_and_names_the_reason():
    async def broken(v):
        raise RuntimeError("403 handshake")

    async def good(v):
        return "OK"

    chain = ProviderChain("tts", [Provider("broken", broken),
                                  Provider("good", good)])
    res = await chain.call("x")
    assert res.value == "OK" and res.provider == "good"
    assert "broken" in res.downgrades[0] and "403" in res.downgrades[0]


@pytest.mark.asyncio
async def test_an_empty_result_is_a_failure_not_a_success():
    """'Whisper empty' once hid a NameError for two hours (18.11.2026)."""
    async def empty(v):
        return ""

    async def good(v):
        return "OK"

    chain = ProviderChain("stt", [Provider("empty", empty), Provider("good", good)])
    res = await chain.call("x")
    assert res.provider == "good"
    assert "empty result" in res.downgrades[0]


@pytest.mark.asyncio
async def test_a_provider_is_cooled_down_after_repeated_failures():
    attempts = []

    async def broken(v):
        attempts.append(v)
        raise RuntimeError("down")

    async def good(v):
        return "OK"

    chain = ProviderChain("tts", [Provider("broken", broken, max_failures=1,
                                           cooldown_s=60),
                                  Provider("good", good)])
    await chain.call("a")
    res = await chain.call("b")
    assert res.provider == "good"
    assert any("cooling" in d for d in res.downgrades)
    assert attempts == ["a"]            # the dead provider was NOT tried again
    assert chain.status()["broken"]["available"] is False


@pytest.mark.asyncio
async def test_all_providers_failing_raises_with_context():
    async def broken(v):
        raise RuntimeError("down")

    chain = ProviderChain("stt", [Provider("a", broken), Provider("b", broken)])
    with pytest.raises(RuntimeError, match="all providers failed"):
        await chain.call("x")


def test_no_providers_is_a_configuration_error():
    with pytest.raises(ValueError):
        ProviderChain("empty", [])
