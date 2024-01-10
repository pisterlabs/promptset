import json
import textwrap
from datetime import datetime

import pytest
from dateutil import tz
from langchain.docstore.document import Document

from chatiq.utils import (
    extract_emoji_text,
    format_mention,
    get_emoji_from_timezone_offset,
    get_timezone_offsets,
    pretty_json_dumps,
    subtract_documents,
    utc_to_local,
)


@pytest.mark.parametrize(
    "text, emoji_pattern, expected_output",
    [
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant :heart: Translate the message from English to Japanese.
                :zap: gpt-4.0
                """
            ).strip(),
            ":robot_face:",
            "You are a helpful assistant :heart: Translate the message from English to Japanese.",
        ),
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant :heart: Translate the message from English to Japanese.
                :zap: gpt-4.0
                """
            ).strip(),
            ":heart:",
            None,
        ),
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant.
                Translate the message from English to Japanese.
                :zap: gpt-4.0
                """
            ).strip(),
            ":robot_face:",
            "You are a helpful assistant.\nTranslate the message from English to Japanese.",
        ),
        (
            textwrap.dedent(
                """
                Do not modify the text below.

                :robot_face: You are a helpful assistant :heart: Translate the message from English to Japanese.
                :zap: gpt-4.0
                """
            ).strip(),
            ":robot_face:",
            "You are a helpful assistant :heart: Translate the message from English to Japanese.",
        ),
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant :heart: Translate the message from English to Japanese.
                :zap: gpt-4.0
                """
            ),
            ":zap:",
            "gpt-4.0",
        ),
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant :heart: Translate the message from English to Japanese.

                :zap: gpt-4.0
                """
            ),
            ":zap:",
            "gpt-4.0",
        ),
        (
            textwrap.dedent(
                """
                :robot_face: You are a helpful assistant.
                :zap: gpt-4.0
                """
            ),
            ":non_existent_emoji:",
            None,
        ),
        (
            "No emoji in this text.",
            ":robot_face:",
            None,
        ),
        (
            "",
            ":robot_face:",
            None,
        ),
    ],
)
def test_extract_emoji_text(text, emoji_pattern, expected_output):
    assert extract_emoji_text(text, emoji_pattern) == expected_output


@pytest.mark.parametrize(
    "user_id, expected_output",
    [
        ("U06FKAYEHF", "<@U06FKAYEHF>"),
        ("U1234567890", "<@U1234567890>"),
        ("", ""),
    ],
)
def test_format_mention(user_id: str, expected_output: str):
    if expected_output:
        assert format_mention(user_id) == expected_output
    else:
        with pytest.raises(ValueError):
            format_mention(user_id)


def test_get_timezone_offsets():
    offsets = get_timezone_offsets()

    assert len(offsets) > 0
    assert len(offsets) == len(set(offsets))

    offsets_in_minutes = [
        (int(offset.split(":")[0]) * 60 + int(offset.split(":")[1]))
        if offset[0] != "-"
        else (int(offset.split(":")[0]) * 60 - int(offset.split(":")[1]))
        for offset in offsets
    ]

    assert offsets_in_minutes == sorted(offsets_in_minutes)


@pytest.mark.parametrize(
    "timezone_offset,expected_emoji",
    [
        ("-11:00", ":clock1:"),
        ("-10:30", ":clock130:"),
        ("+00:00", ":clock12:"),
        ("+03:30", ":clock330:"),
        ("+14:00", ":clock2:"),
    ],
)
def test_get_emoji_from_timezone_offset(timezone_offset, expected_emoji):
    """
    Test get_emoji_from_timezone_offset function.

    Args:
        timezone_offset (str): The timezone offset in "+HH:MM" or "-HH:MM" format.
        expected_emoji (str): The expected output emoji.
    """
    assert get_emoji_from_timezone_offset(timezone_offset) == expected_emoji


def test_utc_to_local():
    utc_dt = datetime(2023, 5, 18, 12, 0, 0, tzinfo=tz.UTC)

    local_dt = utc_to_local(utc_dt, "+02:00")
    assert local_dt == datetime(2023, 5, 18, 14, 0, 0, tzinfo=tz.tzoffset(None, 2 * 3600))

    local_dt = utc_to_local(utc_dt, "-07:00")
    assert local_dt == datetime(2023, 5, 18, 5, 0, 0, tzinfo=tz.tzoffset(None, -7 * 3600))

    local_dt = utc_to_local(utc_dt, "+00:00")
    assert local_dt == utc_dt


def test_pretty_json_dumps():
    obj = {
        "content_type": "message",
        "user": "U0JD6RZU6",
        "channel": "C024BE91L",
        "message": "こんにちは、世界！",
        "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
        "timestamp": "2021-08-20T14:37:41.000200+00:00",
    }

    pretty_json = pretty_json_dumps(obj)

    assert "こんにちは、世界！" in pretty_json
    assert json.loads(pretty_json) == obj


@pytest.fixture
def test_subtract_documents_no_overlap():
    doc1 = Document(page_content='{"content": "Test Domain 1"}', metadata={"url": "https://example.com/1"})
    doc2 = Document(page_content='{"content": "Test Domain 2"}', metadata={"url": "https://example.com/2"})
    doc3 = Document(page_content='{"content": "Test Domain 3"}', metadata={"url": "https://example.com/3"})
    doc4 = Document(page_content='{"content": "Test Domain 4"}', metadata={"url": "https://example.com/4"})

    result = subtract_documents([doc1, doc2], [doc3, doc4])

    assert result == [doc1, doc2]


def test_subtract_documents_overlap():
    doc1 = Document(page_content='{"content": "Test Domain 1"}', metadata={"url": "https://example.com/1"})
    doc2 = Document(page_content='{"content": "Test Domain 2"}', metadata={"url": "https://example.com/2"})
    doc3 = Document(page_content='{"content": "Test Domain 3"}', metadata={"url": "https://example.com/3"})
    doc4 = Document(page_content='{"content": "Test Domain 4"}', metadata={"url": "https://example.com/4"})

    result = subtract_documents([doc1, doc2, doc3], [doc2, doc4])

    assert result == [doc1, doc3]


def test_subtract_documents_all_overlap():
    doc1 = Document(page_content='{"content": "Test Domain"}', metadata={"url": "https://example.com/1"})
    doc2 = Document(page_content='{"content": "Test Domain"}', metadata={"url": "https://example.com/1"})

    result = subtract_documents([doc1, doc2], [doc1, doc2])

    assert result == []


def test_subtract_documents_empty_list():
    doc1 = Document(page_content='{"content": "Test Domain"}', metadata={"url": "https://example.com/1"})
    doc2 = Document(page_content='{"content": "Test Domain"}', metadata={"url": "https://example.com/1"})

    docs1 = [doc1, doc2]
    docs2 = []

    result = subtract_documents(docs1, docs2)

    assert result == docs1
