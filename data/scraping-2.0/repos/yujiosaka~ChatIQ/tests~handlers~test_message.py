import uuid
from logging import Logger
from unittest.mock import call

import pytest
from langchain.docstore.document import Document
from slack_bolt import BoltContext
from slack_sdk import WebClient
from weaviate import Client

from chatiq import ChatIQ
from chatiq.database import Database
from chatiq.document_loaders import MessageLoader, UnfurlingLinkLoader
from chatiq.handlers import MessageHandler
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
    mock_client.conversations_info.return_value = {
        "channel": {
            "topic": {"value": ":speech_balloon: Speak like a pirate"},
            "purpose": {"value": ":thermometer: 2.0\n:round_pushpin: +09:00"},
        }
    }
    return mock_client


@pytest.fixture
def mock_context(mocker):
    mock_context = mocker.MagicMock(spec=BoltContext)
    return mock_context


@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock(spec=Logger)


@pytest.fixture
def mock_say(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_message_loader_init(mocker):
    return mocker.patch.object(MessageLoader, "__init__", return_value=None)


@pytest.fixture
def mock_message_loader_load(mocker):
    mock_message_loader_load = mocker.patch.object(MessageLoader, "load")
    return mock_message_loader_load


@pytest.fixture
def mock_unfurling_link_loader_init(mocker):
    return mocker.patch.object(UnfurlingLinkLoader, "__init__", return_value=None)


@pytest.fixture
def mock_unfurling_link_loader_load(mocker):
    mock_unfurling_link_loader_load = mocker.patch.object(UnfurlingLinkLoader, "load")
    return mock_unfurling_link_loader_load


@pytest.fixture
def mock_vectorstore(mocker):
    mock_vectorstore = mocker.MagicMock(spec=Vectorstore)
    mocker.patch("chatiq.handlers.message.Vectorstore", return_value=mock_vectorstore)
    return mock_vectorstore


@pytest.fixture
def mock_team(mocker):
    mock_team = mocker.MagicMock(spec=SlackTeam)
    mock_team.namespace_uuid = uuid.UUID("b63dfe06-5e32-4fe4-9dda-f0426eb8d83a")
    return mock_team


@pytest.fixture
def mock_repository(mocker, mock_team):
    mock_repository = mocker.MagicMock(spec=SlackTeamRepository)
    mocker.patch("chatiq.handlers.message.SlackTeamRepository", return_value=mock_repository)
    mock_repository.get_or_create.return_value = mock_team
    return mock_repository


def test_message_handler_call_add_event(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "user": "U0JD6RS3T",
            "text": "Hello, World!",
            "ts": "1579244331.000200",
            "thread_ts": "1579244331.000200",
            "channel_type": "channel",
            "channel": "C024BE91L",
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_not_called()
    mock_unfurling_link_loader_load.assert_not_called()
    mock_vectorstore.add_documents.assert_called_once_with(
        mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]
    )
    mock_vectorstore.delete_file_or_attachment.assert_not_called()


def test_message_handler_call_change_event(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "subtype": "message_changed",
            "message": {
                "user": "U0JD6RS3T",
                "text": "こんにちは、世界！",
                "ts": "1579244331.000200",
                "thread_ts": "1579244331.000200",
                "channel_type": "channel",
                "channel": "C024BE91L",
            },
            "previous_message": {
                "user": "U0JD6RS3T",
                "text": "Hello, World!",
                "ts": "1579244331.000200",
                "thread_ts": "1579244331.000200",
                "channel_type": "channel",
                "channel": "C024BE91L",
            },
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"]["message"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_not_called()
    mock_unfurling_link_loader_load.assert_not_called()
    mock_vectorstore.add_documents.assert_called_once_with(
        mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]
    )
    mock_vectorstore.delete_file_or_attachment.assert_not_called()


def test_message_handler_call_add_event_with_plain_text_file(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                    "files": [
                        {
                            "content_type": "python",
                            "title": "Test Python File",
                            "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.py",
                        }
                    ],
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "user": "U0JD6RS3T",
            "text": "Hello, World!",
            "ts": "1579244331.000200",
            "thread_ts": "1579244331.000200",
            "channel_type": "channel",
            "channel": "C024BE91L",
            "files": [
                {
                    "id": "F0JD6RZU6",
                    "filetype": "python",
                    "name": "test.py",
                    "title": "Test Python File",
                    "content": "print('Hello, world!')",
                    "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.py",
                    "timestamp": 1629470261.000200,
                }
            ],
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_not_called()
    mock_unfurling_link_loader_load.assert_not_called()
    mock_vectorstore.add_documents.assert_called_once_with(
        mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]
    )
    mock_vectorstore.delete_file_or_attachment.assert_not_called()


def test_message_handler_call_add_event_with_pdf_file(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                    "files": [
                        {
                            "content_type": "pdf",
                            "title": "parsed pdf content",
                            "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.pdf",
                        }
                    ],
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "user": "U0JD6RS3T",
            "text": "Hello, World!",
            "ts": "1579244331.000200",
            "thread_ts": "1579244331.000200",
            "channel_type": "channel",
            "channel": "C024BE91L",
            "files": [
                {
                    "id": "F0JD6RZU6",
                    "filetype": "pdf",
                    "name": "test.pdf",
                    "title": "Test PDF File",
                    "permalink": "https://chatiq.slack.com/files/U0JD6RZU6/F0JD6RZU6/test.pdf",
                    "timestamp": 1629470261.0002,
                    "url_private": "http://example.com/test.pdf",
                }
            ],
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_not_called()
    mock_unfurling_link_loader_load.assert_not_called()
    mock_vectorstore.add_documents.assert_called_once_with(
        mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]
    )
    mock_vectorstore.delete_file_or_attachment.assert_not_called()


def test_message_handler_call_add_event_with_unfurling_links(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                    "attachments": [
                        {
                            "url": "https://example.com",
                            "title": "Example Domain",
                            "text": "This domain is for use in illustrative examples in documents.",
                        }
                    ],
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]
    mock_unfurling_link_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "unfurling_link",
                    "user": "U0JD6RZU6",
                    "title": "Example Domain",
                    "channel": "C024BE91L",
                    "content": "This domain is for use in illustrative examples in documents.",
                    "page": "1 / 1",
                    "permalink": "https://example.com",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "1",
                "content_type": "unfurling_link",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "0000000000.000000",
                "ts": "1629470261.000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
                "permalink": "https://example.com",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "user": "U0JD6RS3T",
            "text": "Hello, World!",
            "ts": "1579244331.000200",
            "thread_ts": "1579244331.000200",
            "channel_type": "channel",
            "channel": "C024BE91L",
            "attachments": [
                {
                    "id": 1,
                    "original_url": "https://example.com",
                    "title": "Example Domain",
                    "text": "This domain is for use in illustrative examples in documents.",
                }
            ],
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_called_once_with(
        body, body["event"], body["event"]["attachments"][0], mock_team.model
    )
    mock_unfurling_link_loader_load.assert_called_once()
    calls = [
        call(mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]),
        call(mock_unfurling_link_loader_load.return_value),
    ]
    mock_vectorstore.add_documents.assert_has_calls(calls)
    mock_vectorstore.delete_file_or_attachment.assert_not_called()


def test_message_handler_call_change_event_with_unfurling_links(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_message_loader_init,
    mock_message_loader_load,
    mock_unfurling_link_loader_init,
    mock_unfurling_link_loader_load,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    mock_message_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "message",
                    "user": "U0JD6RZU6",
                    "channel": "C024BE91L",
                    "message": "Hello, world!",
                    "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "Ev058XH6RPGR",
                "content_type": "message",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "1579244331.000200",
                "ts": "1579244331.000200",
                "permalink": "https://chatiq.slack.com/archives/C024BE91L/p1579244331000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
            },
        )
    ]
    mock_unfurling_link_loader_load.return_value = [
        Document(
            page_content=pretty_json_dumps(
                {
                    "content_type": "unfurling_link",
                    "user": "U0JD6RZU6",
                    "title": "Example Domain",
                    "channel": "C024BE91L",
                    "content": "This domain is for use in illustrative examples in documents.",
                    "page": "1 / 1",
                    "permalink": "https://example.com",
                    "timestamp": "2021-08-20T14:37:41.000200+00:00",
                },
            ),
            metadata={
                "file_or_attachment_id": "1629470261.000200-1",
                "content_type": "unfurling_link",
                "channel_type": "channel",
                "channel_id": "C024BE91L",
                "thread_ts": "0000000000.000000",
                "ts": "1629470261.000200",
                "timestamp": "2021-08-20T14:37:41.000200+00:00",
                "permalink": "https://example.com",
            },
        )
    ]

    message_handler = MessageHandler(mock_chatiq)
    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "subtype": "message_changed",
            "message": {
                "user": "U0JD6RS3T",
                "text": "Hello, World!",
                "ts": "1579244331.000200",
                "thread_ts": "1579244331.000200",
                "channel_type": "channel",
                "channel": "C024BE91L",
            },
            "previous_message": {
                "user": "U0JD6RS3T",
                "text": "Hello, World!",
                "ts": "1579244331.000200",
                "thread_ts": "1579244331.000200",
                "channel_type": "channel",
                "channel": "C024BE91L",
                "attachments": [
                    {
                        "id": 1,
                        "original_url": "https://example.com",
                        "title": "Example Domain",
                        "text": "This domain is for use in illustrative examples in documents.",
                    }
                ],
            },
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_message_loader_init.assert_called_once_with(mock_client, body, body["event"]["message"], mock_team.model)
    mock_message_loader_load.assert_called_once()
    mock_unfurling_link_loader_init.assert_called_once_with(
        body, body["event"]["previous_message"], body["event"]["previous_message"]["attachments"][0], mock_team.model
    )
    mock_unfurling_link_loader_load.assert_called_once()
    mock_vectorstore.add_documents.assert_called_once_with(
        mock_message_loader_load.return_value, uuids=[uuid.UUID("1e86735d-2b15-5ca1-be34-0142abd47742")]
    )
    mock_vectorstore.delete_file_or_attachment.assert_called_once_with("1629470261.000200-1")


def test_message_handler_call_channel_info_event(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
):
    message_handler = MessageHandler(mock_chatiq)

    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "subtype": "channel_topic",
            "channel": "C024BE91L",
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_client.conversations_info.assert_called_once_with(channel="C024BE91L")
    mock_say.assert_called_once_with(
        text="Configuration is set for this channel.",
        blocks=[
            {"type": "section", "text": {"text": "Configuration is set for this channel", "type": "plain_text"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "*AI temperature:*\n2.0"},
                    {"type": "mrkdwn", "text": "*Timezone:*\n+09:00"},
                ],
            },
            {"type": "section", "text": {"type": "mrkdwn", "text": "*System Message*\nSpeak like a pirate"}},
        ],
    )


def test_message_handler_call_delete_event(
    mock_chatiq,
    mock_client,
    mock_context,
    mock_logger,
    mock_say,
    mock_vectorstore,
):
    message_handler = MessageHandler(mock_chatiq)

    body = {
        "team_id": "T0JD6RZU6",
        "event": {
            "subtype": "message_deleted",
            "previous_message": {
                "ts": "1579244331.000200",
                "user": "U0JD6RS3T",
                "text": "Hello, World!",
                "thread_ts": "1579244331.000200",
                "channel_type": "channel",
                "channel": "C024BE91L",
            },
        },
        "authorizations": [{"user_id": "U0JD6RZU6"}],
    }
    message_handler(mock_client, mock_context, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_vectorstore.ensure_index.assert_called_once()
    mock_vectorstore.delete_message.assert_called_once_with("1579244331.000200")
