import pytest
from main import is_valid_text, WHISPER_HALLUCINATIONS, SINGLE_WORD_HALLUCINATIONS

def test_is_valid_text_empty_or_punctuation():
    assert is_valid_text("") is False
    assert is_valid_text("   ") is False
    assert is_valid_text(".,?!-") is False

def test_is_valid_text_short_valid():
    assert is_valid_text("hello") is True
    assert is_valid_text("hi there") is True

def test_is_valid_text_repeating_characters():
    assert is_valid_text("haaaaaaaaaaa") is False
    assert is_valid_text("aaabbbcccdd") is True

def test_is_valid_text_single_word_hallucinations():
    for word in SINGLE_WORD_HALLUCINATIONS:
        assert is_valid_text(word) is False
    assert is_valid_text("word") is True
    assert is_valid_text("ok") is False

def test_is_valid_text_whisper_hallucinations():
    for hall in WHISPER_HALLUCINATIONS:
        assert is_valid_text(hall) is False
        assert is_valid_text(f"prefix {hall} suffix") is False

def test_is_valid_text_normal():
    assert is_valid_text("This is a completely valid text.") is True
