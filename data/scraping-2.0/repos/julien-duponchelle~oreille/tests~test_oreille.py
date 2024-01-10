#!/usr/bin/env python

"""Tests for `oreille` package."""

import pytest
import pydub
import openai
from openai.openai_object import OpenAIObject
from unittest.mock import ANY

from oreille import oreille


@pytest.fixture
def empty_audio(tmp_path, scope="module"):
    """
    An empty audio file
    """
    with open(tmp_path / "empty.wav", "wb+") as f:
        empty_audio = pydub.AudioSegment.silent(1)
        empty_audio.export(f, format="wav")
    return tmp_path / "empty.wav"


@pytest.fixture
def long_empty_audio(tmp_path, scope="module"):
    """
    An audio file of 11 minutes
    """
    with open(tmp_path / "long_empty.wav", "wb+") as f:
        empty_audio = pydub.AudioSegment.silent(11 * 60 * 1000)
        empty_audio.export(f, format="wav")
    return tmp_path / "long_empty.wav"


@pytest.fixture
def openai_object():
    result = OpenAIObject(response_ms=100)
    result.text = "Hello World"
    result.segments = [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.8000000000000003,
            "text": "Hello",
            "tokens": [50364, 8257, 53],
            "temperature": 0.0,
            "avg_logprob": -0.32106150751528534,
            "compression_ratio": 0.9402985074626866,
            "no_speech_prob": 0.03250369429588318,
        },
        {
            "id": 1,
            "seek": 0,
            "start": 2.9,
            "end": 3.5,
            "text": "world",
            "tokens": [50364, 8257, 53],
            "temperature": 0.0,
            "avg_logprob": -0.32106150751528534,
            "compression_ratio": 0.9402985074626866,
            "no_speech_prob": 0.03250369429588318,
        },
    ]
    return result


@pytest.fixture
def openai_object2():
    result = OpenAIObject(response_ms=200)
    result.text = "Bonjour le monde"
    result.segments = [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.8000000000000003,
            "text": "Bonjour",
            "tokens": [50364, 8257, 53],
            "temperature": 0.0,
            "avg_logprob": -0.32106150751528534,
            "compression_ratio": 0.9402985074626866,
            "no_speech_prob": 0.03250369429588318,
        },
        {
            "id": 1,
            "seek": 0,
            "start": 2.9,
            "end": 3.5,
            "text": "le monde",
            "tokens": [50364, 8257, 53],
            "temperature": 0.0,
            "avg_logprob": -0.32106150751528534,
            "compression_ratio": 0.9402985074626866,
            "no_speech_prob": 0.03250369429588318,
        },
    ]
    return result


def test_transcribe_verbose_json(mocker, empty_audio, openai_object):
    mocker.patch("openai.Audio.transcribe", return_value=openai_object)

    transcribe = oreille.transcribe(
        "whisper-1", empty_audio, response_format="verbose_json"
    )

    openai.Audio.transcribe.assert_called_with(
        "whisper-1", ANY, response_format="verbose_json"
    )
    assert transcribe.text == "Hello World"
    assert transcribe.response_ms == 100
    assert transcribe.segments == openai_object.segments


def test_transcribe_verbose_json_long(
    mocker, long_empty_audio, openai_object, openai_object2
):
    mocker.patch("openai.Audio.transcribe", side_effect=[openai_object, openai_object2])

    transcribe = oreille.transcribe(
        "whisper-1", long_empty_audio, response_format="verbose_json"
    )

    openai.Audio.transcribe.assert_called_with(
        "whisper-1", ANY, response_format="verbose_json"
    )
    assert transcribe.text == "Hello World Bonjour le monde"
    assert transcribe.response_ms == 300
    assert len(transcribe.segments) == 4
    assert transcribe.segments[0]["id"] == 0
    assert transcribe.segments[0]["start"] == 0
    assert transcribe.segments[0]["end"] == 2.8000000000000003
    assert transcribe.segments[2]["id"] == 2
    assert transcribe.segments[2]["start"] == 600
    assert transcribe.segments[2]["end"] == 602.8
    assert transcribe.segments[3]["id"] == 3
    assert transcribe.segments[3]["start"] == 602.9
    assert transcribe.segments[3]["end"] == 603.5


def test_transcribe_text(mocker, empty_audio, openai_object):
    mocker.patch("openai.Audio.transcribe", return_value=openai_object)

    transcribe = oreille.transcribe("whisper-1", empty_audio, response_format="text")

    openai.Audio.transcribe.assert_called_with(
        "whisper-1", ANY, response_format="verbose_json"
    )
    assert transcribe == "Hello World"


def test_transcribe_text_long(mocker, long_empty_audio):
    o1 = OpenAIObject(response_ms=100)
    o1.text = "Hello World"
    o2 = OpenAIObject(response_ms=200)
    o2.text = "Bonjour le monde"
    mocker.patch("openai.Audio.transcribe", side_effect=[o1, o2])

    transcribe = oreille.transcribe(
        "whisper-1", long_empty_audio, response_format="text"
    )

    assert transcribe == "Hello World Bonjour le monde"
