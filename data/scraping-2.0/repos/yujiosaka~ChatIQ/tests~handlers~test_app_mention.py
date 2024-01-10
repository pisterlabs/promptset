import uuid
from logging import Logger

import pytest
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from slack_sdk import WebClient
from weaviate import Client

from chatiq import ChatIQ
from chatiq.chat_chain import ChatChain
from chatiq.database import Database
from chatiq.handlers import AppMentionHandler
from chatiq.models import SlackTeam
from chatiq.repositories import SlackTeamRepository
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
            "topic": {"value": ":thermometer: 2.0"},
            "purpose": {"value": ":round_pushpin: +09:00"},
            "is_private": False,
        }
    }
    mock_client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U0JD6RS3T",
                "text": "Hello, AI!",
                "ts": "1652944800.000000",
            },
            {
                "user": "U0JD6RZU6",
                "text": "Hello, I'm an AI!",
                "ts": "1652944900.000000",
            },
            {
                "user": "U0JD6RS3T",
                "text": "Help me, AI!",
                "ts": "1652945000.000000",
            },
        ]
    }
    return mock_client


@pytest.fixture
def mock_say(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock(spec=Logger)


@pytest.fixture
def mock_chat_chain(mocker):
    mock_chat_chain = mocker.MagicMock(spec=ChatChain)
    mocker.patch("chatiq.handlers.app_mention.ChatChain", return_value=mock_chat_chain)
    return mock_chat_chain


@pytest.fixture
def mock_chat_openai(mocker):
    mock_chat_openai = mocker.MagicMock(spec=ChatOpenAI)
    mocker.patch("chatiq.handlers.app_mention.ChatOpenAI", return_value=mock_chat_openai)
    return mock_chat_openai


@pytest.fixture
def mock_memory(mocker):
    mock_memory = mocker.MagicMock(spec=ConversationTokenBufferMemory)
    mocker.patch("chatiq.handlers.app_mention.ConversationTokenBufferMemory", return_value=mock_memory)
    return mock_memory


@pytest.fixture
def mock_vectorstore(mocker):
    mock_vectorstore = mocker.MagicMock(spec=Vectorstore)
    mocker.patch("chatiq.handlers.app_mention.Vectorstore", return_value=mock_vectorstore)
    return mock_vectorstore


@pytest.fixture
def mock_team(mocker):
    mock_team = mocker.MagicMock(spec=SlackTeam)
    mock_team.model = "gpt-3.5-turbo"
    mock_team.namespace_uuid = uuid.UUID("b63dfe06-5e32-4fe4-9dda-f0426eb8d83a")
    return mock_team


@pytest.fixture
def mock_repository(mocker, mock_team):
    mock_repository = mocker.MagicMock(spec=SlackTeamRepository)
    mocker.patch("chatiq.handlers.app_mention.SlackTeamRepository", return_value=mock_repository)
    mock_repository.get_or_create.return_value = mock_team
    return mock_repository


def test_app_mention_handler_call(
    mock_chatiq,
    mock_client,
    mock_logger,
    mock_say,
    mock_chat_openai,
    mock_memory,
    mock_chat_chain,
    mock_vectorstore,
    mock_repository,
    mock_team,
):
    handler = AppMentionHandler(mock_chatiq)
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

    handler(mock_client, body, mock_logger, mock_say)
    for thread in mock_chatiq.threads:
        thread.join()

    mock_client.conversations_info.assert_called_once_with(channel="C024BE91L")
    mock_repository.get_or_create.assert_called_once_with(body["team_id"], body["authorizations"][0]["user_id"])
    mock_vectorstore.ensure_index.assert_called_once()
    mock_vectorstore.as_retriever.assert_called_once_with(False, "C024BE91L", "1579244331.000200")
    mock_client.conversations_replies.assert_called_once_with(channel="C024BE91L", ts="1579244331.000200")
    mock_chat_chain.add_memory_ai_message.assert_called_once_with(
        {"user": "U0JD6RZU6", "text": "Hello, I'm an AI!", "ts": "1652944900.000000"}
    )
    mock_chat_chain.run.assert_called_once_with(
        {
            "user": "U0JD6RS3T",
            "text": "Help me, AI!",
            "ts": "1652945000.000000",
        }
    )
    mock_say.assert_called_once_with(mock_chat_chain.run.return_value, thread_ts="1579244331.000200")
