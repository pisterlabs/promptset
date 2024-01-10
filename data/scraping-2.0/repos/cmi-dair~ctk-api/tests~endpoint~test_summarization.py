"""Endpoint tests for the summarization router."""
# pylint: disable=redefined-outer-name
import dataclasses
import tempfile

import pytest
import pytest_mock
from fastapi import status, testclient
from pytest_mock import plugin

from . import conftest


@dataclasses.dataclass(frozen=True)
class OpenAiMessage:
    """Represents a message from OpenAI's GPT.

    Attributes:
        content (str): The content of the message.
    """

    content: str = "test message"


@dataclasses.dataclass(frozen=True)
class OpenAiChoice:
    """Represents a choice in the GPT response..

    Attributes:
        message (OpenAiMessage): The message associated with the choice.
    """

    message: OpenAiMessage = dataclasses.field(default_factory=OpenAiMessage)


@dataclasses.dataclass(frozen=True)
class OpenAiResponse:
    """Represents a response from the OpenAI API.

    Attributes:
        choices (list[OpenAiChoice]): The list of choices in the response.
    """

    choices: list[OpenAiChoice] = dataclasses.field(
        default_factory=lambda: [OpenAiChoice()],
    )


@pytest.fixture(autouse=True)
def _mock_openai_response(mocker: plugin.MockerFixture) -> None:
    """Returns a mock OpenAI response."""
    response = OpenAiResponse()
    openai = mocker.MagicMock()
    mocker.patch("openai.OpenAI", return_value=openai)
    openai.chat = mocker.MagicMock()
    openai.chat.completions = mocker.MagicMock()
    openai.chat.completions.create = mocker.MagicMock(return_value=response)


def test_anonymization_endpoint(
    client: testclient.TestClient,
    endpoints: conftest.Endpoints,
    document: tempfile._TemporaryFileWrapper,
) -> None:
    """Tests the anonymization endpoint."""
    form_data = {"docx_file": document}
    expected = (
        "clinical summary and impressions\nName: [FIRST_NAME] [LAST_NAME]\n"
        "He/She he/she himself/herself man/woman"
    )

    response = client.post(endpoints.ANONYMIZE_REPORT, files=form_data)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected


def test_summarization_endpoint_new(
    mocker: plugin.MockerFixture,
    client: testclient.TestClient,
    endpoints: conftest.Endpoints,
) -> None:
    """Tests the summarization endpoint."""
    mocker.patch(
        "ctk_api.routers.summarization.controller._check_for_existing_document",
        return_value=None,
    )

    response = client.post(endpoints.SUMMARIZE_REPORT, json={"text": "Hello there."})

    assert response.status_code == status.HTTP_201_CREATED
    assert (
        response.headers["Content-Type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert (
        response.headers["Content-Disposition"] == 'attachment; filename="summary.docx"'
    )


def test_summarization_endpoint_exists(
    mocker: pytest_mock.MockFixture,
    client: testclient.TestClient,
    endpoints: conftest.Endpoints,
) -> None:
    """Tests the summarization endpoint when the document already exists."""
    document = {"summary": "Hello there."}
    mocker.patch(
        "ctk_api.routers.summarization.controller._check_for_existing_document",
        return_value=document,
    )

    response = client.post(endpoints.SUMMARIZE_REPORT, json={"text": "Hello there."})

    assert response.status_code == status.HTTP_200_OK
    assert (
        response.headers["Content-Type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert (
        response.headers["Content-Disposition"] == 'attachment; filename="summary.docx"'
    )
