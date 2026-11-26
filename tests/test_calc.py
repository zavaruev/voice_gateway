"""L1 calculator: spoken arithmetic and percentages, no LLM, no network."""

import importlib.util
import json
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import calc  # noqa: E402
import resolver  # noqa: E402


# --- expression evaluation ---------------------------------------------------

def test_addition_in_words():
    assert calc.evaluate("два плюс два") == 4.0
    assert calc.evaluate("сто пятьдесят плюс двадцать пять") == 175.0


def test_standard_precedence_and_parentheses():
    assert calc.evaluate("2 + 3 * 4") == 14.0
    assert calc.evaluate("(2 + 3) * 4") == 20.0
    assert calc.evaluate("10 - 2 - 3") == 5.0


def test_decimal_comma_and_division():
    assert calc.evaluate("1,5 + 2,5") == 4.0
    assert calc.evaluate("10 разделить на 4") == 2.5
    assert calc.evaluate("9 / 2") == 4.5


def test_percent_of():
    assert calc.evaluate("15 процентов от 200") == 30.0
    assert calc.evaluate("2% от 150") == 3.0
    assert calc.evaluate("двести процентов от 40") == 80.0


def test_power_and_root():
    assert calc.evaluate("2 в степени 3") == 8.0
    assert calc.evaluate("2^3") == 8.0
    assert calc.evaluate("корень из 16") == 4.0
    assert calc.evaluate("квадратный корень из 16") == 4.0


def test_recipe_units_and_and_as_addition():
    """Kitchen phrasings from 23.11.2026: the calculator must not die on a
    gram or a milliliter after the number, and «2 и 3» means 2 + 3."""
    assert calc.evaluate("10% от 200 граммов") == 20.0
    assert calc.evaluate("200 г плюс 20 г") == 220.0
    assert calc.evaluate("двести граммов плюс пятьдесят") == 250.0
    assert calc.evaluate("сложи 2 и 3") == 5.0
    assert calc.evaluate("100 граммов умножить на 2") == 200.0


def test_invalid_and_unsafe_input_is_refused():
    assert calc.evaluate("привет") is None
    assert calc.evaluate("2 +") is None
    assert calc.evaluate("__import__('os')") is None
    assert calc.evaluate("10 / 0") is None


def test_spoken_sentence_uses_digits_for_the_tts_choke_point():
    assert calc.sentence(4.0) == "Получается 4."
    assert calc.sentence(2.5) == "Получается 2,5."


# --- routing gate ------------------------------------------------------------

def test_calc_phrases_are_recognised():
    assert resolver.resolve_calc("два плюс два").value == 4.0
    assert resolver.resolve_calc("посчитай 15 процентов от 200").value == 30.0
    assert resolver.resolve_calc("сколько будет корень из 16").value == 4.0


def test_existing_home_questions_are_not_stolen():
    for text in (
        "сколько времени",
        "сколько процентов заряда пылесоса",
        "сделай яркость 20 процентов",
        "что такое корень из трех",
        "два процента от того что на весах",
        "сколько весит",
        "включи свет",
    ):
        assert resolver.resolve_calc(text) is None, text


# --- SSE integration ---------------------------------------------------------

def _router_app():
    spec = importlib.util.spec_from_file_location(
        "jev_router_calc_app", os.path.join(_ROUTER, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calc_route_answers_without_the_classifier(monkeypatch):
    app = _router_app()

    def _must_not_run(*args, **kwargs):
        raise AssertionError("classifier must not run for an expression")

    monkeypatch.setattr(app.classifier, "classify", _must_not_run)

    async def _drive():
        gen = app._handle(app.RouteRequest(
            text="два плюс два", session_id="probe", stream_name="kitchen"
        ))
        first = await gen.__anext__()
        second = await gen.__anext__()
        await gen.aclose()
        return json.loads(first.removeprefix("data: ")), json.loads(
            second.removeprefix("data: ")
        )

    route, sentence = __import__("asyncio").run(_drive())
    assert route["route"] == "calc"
    assert route["confidence"] == 1.0
    assert sentence["text"] == "Получается 4."


def test_the_scale_percentage_still_wins_over_calculator(monkeypatch):
    app = _router_app()

    async def _drive():
        gen = app._handle(app.RouteRequest(
            text="два процента от того что на весах",
            session_id="probe", stream_name="kitchen",
        ))
        first = await gen.__anext__()
        await gen.aclose()
        return json.loads(first.removeprefix("data: "))

    route = __import__("asyncio").run(_drive())
    assert route["route"] == "scale"
    assert route["reason"] == "scale_percent"
