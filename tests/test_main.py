import json
from unittest import mock

from main import load_speaker_names

def test_load_speaker_names_success():
    mock_data = '{"1": "John", "2": "Jane"}'
    with mock.patch('builtins.open', mock.mock_open(read_data=mock_data)):
        result = load_speaker_names()
        assert result == {"1": "John", "2": "Jane"}

def test_load_speaker_names_file_not_found():
    with mock.patch('builtins.open', side_effect=FileNotFoundError):
        result = load_speaker_names()
        assert result == {}

def test_load_speaker_names_invalid_json():
    mock_data = 'invalid json'
    with mock.patch('builtins.open', mock.mock_open(read_data=mock_data)):
        result = load_speaker_names()
        assert result == {}
