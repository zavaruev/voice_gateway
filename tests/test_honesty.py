"""Honesty veto tests: an action claim without a confirmed ha_action must
never be spoken (field case: «Кафеварка включена!» after MatchFailedError).

Runs offline on host python: honesty.py is pure stdlib, no smolagents.
"""

import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
sys.path.insert(0, _WORKER)

from honesty import vet_answer  # noqa: E402

_FAIL_NAME = [
    {"tool": "intent__HassTurnOn", "ok": False,
     "detail": "Тул вернул ошибку: MatchFailedReason.NAME: 1, name='кафеварка'"}
]
_FAIL_ASSISTANT = [
    {"tool": "intent__HassTurnOn", "ok": False,
     "detail": "Тул вернул ошибку: MatchFailedReason.ASSISTANT: 8"}
]
_OK = [{"tool": "intent__HassTurnOn", "ok": True, "detail": '{"success": true}'}]


def test_veto_replaces_claim_after_failed_action():
    """The exact field regression: claim + only failed actions -> truth."""
    out, replaced = vet_answer("Кафеварка включена!", _FAIL_NAME)
    assert replaced is True
    assert "Не нашла такого устройства" in out


def test_veto_keeps_honest_failure():
    out, replaced = vet_answer("Не получилось включить, устройство не найдено.",
                               _FAIL_NAME)
    assert replaced is False
    assert out == "Не получилось включить, устройство не найдено."


def test_no_veto_when_action_succeeded():
    out, replaced = vet_answer("Кафеварка включена!", _OK)
    assert replaced is False and out == "Кафеварка включена!"


def test_no_veto_without_recorded_events():
    """No ha_action at all (e.g. only ha_read) — nothing to prove."""
    out, replaced = vet_answer("Кафеварка включена!", [])
    assert replaced is False


def test_success_after_earlier_failure_passes():
    events = _FAIL_NAME + _OK
    out, replaced = vet_answer("Включил!", events)
    assert replaced is False


def test_unexposed_device_message():
    out, replaced = vet_answer("Готово, включил!", _FAIL_ASSISTANT)
    assert replaced is True and "ассистент" in out


def test_non_action_answers_untouched():
    out, replaced = vet_answer("Расскажу анекдот: заходит кактус...", _FAIL_NAME)
    assert replaced is False
