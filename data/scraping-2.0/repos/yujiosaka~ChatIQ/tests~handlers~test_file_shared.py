import uuid
from logging import Logger

import pytest
from langchain.docstore.document import Document
from slack_bolt import BoltContext
from slack_sdk import WebClient
from weaviate import Client

from chatiq import ChatIQ
from chatiq.database import Database
from chatiq.document_loaders import PdfLoader, PlainTextLoader
from chatiq.handlers import FileSharedHandler
from chatiq.models import SlackTeam
from chatiq.repositories import SlackTeamRepository
from chatiq.utils import pretty_json_dumps
from chatiq.vectorstore import Vectorstore


@pytest.fixture
def mock_chatiq(mocker):
    mock_chatiq = mocker.MagicMock(spec=ChatIQ)
    mock_chatiq.weaviate_client = mocker.MagicMock(spec=Client)
    mock_chatiq.db = mocker.MagicMock(spec=Database)
    mock_chatiq.threads = []
    mock_chatiq.add_thread.side_effect = lambda thread: mock_chatiq.threads.append(thread)
    return mock_chatiq


@pytest.fixture
def mock_client(mocker):
    mock_client = mocker.MagicMock(spec=WebClient)
    return mock_client


@pytest.fixture
def mock_context(mocker):
    mock_context = mocker.MagicMock(spec=BoltContext)
    return mock_context


@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock(spec=Logger)


@pytest.fixture
def mock_plain_text_loader_init(mocker):
    return mocker.patch.object(PlainTextLoader, "__init__", return_value=None)


@pytest.fixture
def mock_plain_text_loader_load(mocker):
    mock_plain_text_loader_load = mocker.patch.object(PlainTextLoader, "load")
    return mock_plain_text_loader_load


@pytest.fixture
def mock_pdf_loader_init(mocker):
    return mocker.patch.object(PdfLoader, "__init__", return_value=None)


@pytest.fixture
def mock_pdf_loader_load(mocker):
    mock_pdf_loader_load = mocker.patch.object(PdfLoader, "load")
    return mock_pdf_loader_load


@pytest.fixture
def mock_vectorstore(mocker):
    mock_vectorstore = mocker.MagicMock(spec=Vectorstore)
    mocker.patch("chatiq.handlers.file_shared.Vectorstore", return_value=mock_vectorstore)
    return mock_vectorstore


@pytest.fixture
def mock_team(mocker):
    mock_team = mocker.MagicMock(spec=SlackTeam)
    mock_team.namespace_uuid = uuid.UUID("b63dfe06-5e32-4fe4-9dda-f0426eb8d83a")
    return mock_team


@pytest.fixture
def mock_repository(mocker, mock_team):
    mock_repository = mocker.MagicMock(spec=SlackTeamRepository)
    mocker.patch("chatiq.handlers.file_shared.SlackTeamRepository", return_value=mock_repository)
    mock_repository.get_or_create.return_value = mock_team
    return mock_repository


def test_file_shared_handler_call_plain_text_file(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_plain_text_loader_init,
    mock_plain_text_loader_load,
    mock_pdf_loader_init,
    mock_pdf_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_client.files_info.return_value = {
        "content": "print('Hello, world!')",
        "file": {
            "id": "F0JD6RZU6",
            "filetype": "python",
            "name": "test.py",
            "title": "Test Python File",
            "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.py",
        },
    }
    mock_client.conversations_info.return_value = {
        "channel": {
            "is_channel": True,
            "is_group": False,
            "is_im": False,
            "is_mpim": False,
        },
    }
    mock_plain_text_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "python",
                    "user": "U0JD6RZU6",
                    "name": "test.py",
                    "title": "Test Python File",
                    "channel": "C024BE91L",
                    "content": "print('Hello, world!')",
                    "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.py",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "F0JD6RZU6",
                "content_type": "python",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "0000000000.000000",
                "ts": "1629470261.000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
                "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.py",
                "title": "Test Python File",
            },
        )
    ]

    file_shared_handler = FileSharedHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "file_id": "F0JD6RZU6",
            "channel_id": "C024BE91L",
            "user_id": "U0JD6RZU6",
            "event_ts": "1629470261.000200",
        },
        "event_time": 1629470261,
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    file_shared_handler(mock_client, mock_context, body, mock_logger)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_plain_text_loader_init.assert_called_once_with(
        mock_client.files_info.return_value["content"],
        body,
        mock_client.files_info.return_value["file"],
        "channel",
        mock_team.model,
    )
    mock_plain_text_loader_load.assert_called_once()
    mock_pdf_loader_init.assert_not_called()
    mock_pdf_loader_load.assert_not_called()
    mock_vectorstore.add_documents.assert_called_once_with(mock_plain_text_loader_load.return_value)


def test_file_shared_handler_call_pdf_file(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_plain_text_loader_init,
    mock_plain_text_loader_load,
    mock_pdf_loader_init,
    mock_pdf_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_client.files_info.return_value = {
        "content": None,
        "file": {
            "id": "F0JD6RZU6",
            "filetype": "pdf",
            "name": "test.pdf",
            "title": "Test PDF File",
            "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.pdf",
            "url_private": "http://example.com/test.pdf",
        },
    }
    mock_client.conversations_info.return_value = {
        "channel": {
            "is_channel": False,
            "is_group": True,
            "is_im": False,
            "is_mpim": False,
        },
    }
    mock_pdf_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "pdf",
                    "user": "U0JD6RZU6",
                    "name": "test.pdf",
                    "title": "Test PDF File",
                    "channel": "C024BE91L",
                    "content": "parsed pdf content",
                    "page": "1 / 1",
                    "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.pdf",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "F0JD6RZU6",
                "content_type": "pdf",
                "channel_type": "group",
                "channel_id": "C024BE91L",
                "thread_ts": "0000000000.000000",
                "ts": "70261.000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
                "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.pdf",
                "title": "Test PDF File",
            },
        )
    ]

    file_shared_handler = FileSharedHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "file_id": "F0JD6RZU6",
            "channel_id": "C024BE91L",
            "user_id": "U0JD6RZU6",
            "event_ts": "1629470261.000200",
        },
        "event_time": 1629470261,
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    file_shared_handler(mock_client, mock_context, body, mock_logger)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_plain_text_loader_init.assert_not_called()
    mock_plain_text_loader_load.assert_not_called()
    mock_pdf_loader_init.assert_called_once_with(
        mock_context, body, mock_client.files_info.return_value["file"], "group", mock_team.model
    )
    mock_pdf_loader_load.assert_called_once()
    mock_vectorstore.add_documents.assert_called_once_with(mock_pdf_loader_load.return_value)
