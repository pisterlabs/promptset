from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from google.cloud import translate_v2, firestore
from google.cloud.firestore import Client
from google.cloud.firestore_v1.document import DocumentSnapshot

import config
import helpers
import openai
from helpers import increment_message_count, get_user_lang, validate_language, increment_active_chats
from helpers import translate_and_send_messages
from google.api_core.exceptions import FailedPrecondition

# Replace with your actual chat_id and user_id
CHAT_ID = "123456"
USER_ID = "123"

config.MESSAGE_LIMIT = 2

@pytest.mark.asyncio
async def test_increment_message_count_new_chat():
    chat_id = "new_chat"
    assert await increment_message_count(chat_id) == 1

@pytest.mark.asyncio
async def test_increment_message_count_existing_chat():
    chat_id = "existing_chat"
    await increment_message_count(chat_id)
    assert await increment_message_count(chat_id) == 2

@pytest.mark.asyncio
async def test_increment_message_count_multi_chat():
    chat_id1 = "chat1"
    chat_id2 = "chat2"
    assert await increment_message_count(chat_id1) == 1
    assert await increment_message_count(chat_id2) == 1
    assert await increment_message_count(chat_id1) == 2
    assert await increment_message_count(chat_id2) == 2


def test_get_user_lang_existing_user():
    chat_id = "chat1"
    user_id = "user1"
    user_lang = "en"

    client = MagicMock(spec=Client)
    client.collection().document().collection().document().get().exists = True
    client.collection().document().collection().document().get().to_dict.return_value = {"preferred_language": user_lang}
    helpers.db = client
    result = get_user_lang(chat_id, user_id)

    # Uncomment the following line if the function uses the Firestore client directly
    # result = get_user_lang(chat_id, user_id, client)

    assert result == user_lang

def test_get_user_lang_non_existing_user():
    chat_id = "chat1"
    user_id = "non_existing_user"

    client = MagicMock(spec=Client)
    client.collection().document().collection().document().get().exists = False
    helpers.db = client

    result = get_user_lang(chat_id, user_id)

    # Uncomment the following line if the function uses the Firestore client directly
    # result = get_user_lang(chat_id, user_id, client)

    assert result is None

def test_validate_language():
    assert validate_language("en") is True
    assert validate_language("invalid_code") is False

@pytest.mark.asyncio
async def test_get_openai_response():
    response = await openai.get_openai_response("hello again")
    print(response)


@pytest.mark.asyncio
async def test_increment_active_chats_within_limit():
    chat_id = "12345"
    client = MagicMock()
    doc_snapshot = DocumentSnapshot(
        reference=None, data={"count": config.MAXIMUM_CHATS - 1}, exists=True, read_time=None, create_time=None, update_time=None
    )
    # Simulate an active chat count below the limit
    client.collection().document().update.side_effect = [None]
    client.collection().document().get.return_value = doc_snapshot
    helpers.db = client

    result = await increment_active_chats()
    assert result is True

@pytest.mark.asyncio
async def test_increment_active_chats_exceed_limit():
    chat_id = "12345"
    client = MagicMock()
    # Create a fake DocumentSnapshot object
    doc_snapshot = DocumentSnapshot(
        reference=None, data={"count": config.MAXIMUM_CHATS + 1}, exists=True, read_time=None, create_time=None, update_time=None
    )
    # Simulate an active chat count equal to the limit
    client.collection().document().update.side_effect = [None, None]
    client.collection().document().get.return_value = doc_snapshot
    helpers.db = client

    result = await increment_active_chats()
    assert result is False

@pytest.mark.asyncio
async def test_increment_active_chats_failed_precondition():
    chat_id = "12345"
    client = MagicMock()
    # Create a fake DocumentSnapshot object

    # Simulate a FailedPrecondition error
    client.collection().document().update.side_effect = [FailedPrecondition("failed")]
    client.collection().document().set.side_effect = [None]
    helpers.db = client

    result = await increment_active_chats()
    assert result is True



@pytest.mark.asyncio
@patch("helpers.translate_client")
@patch("helpers.db")
async def test_translate_and_send_messages_and_skip_sender(mock_db, mock_translate_client):
    # Set up the mock for the members stream
    mock_members = [
        MagicMock(id="user1", to_dict=lambda: {"preferred_language": "en"}),
        MagicMock(id="user2", to_dict=lambda: {"preferred_language": "fr"}),
        MagicMock(id="user3", to_dict=lambda: {"preferred_language": "es"}),
    ]
    mock_db.collection.return_value.document.return_value.collection.return_value.stream.return_value = mock_members

    # Set up the mock for the translate_client
    translations = [
        {"translatedText": "Translated text in English"},
        {"translatedText": "Texte traduit en français"},
        {"translatedText": "Texto traducido al español"},
    ]
    mock_translate_client.translate.side_effect = translations

    # Set up update and context MagicMock objects
    update = MagicMock(effective_chat=MagicMock(id="chat1"), effective_user=MagicMock(id="user1"), effective_message=MagicMock(message_id="msg1"))
    context = MagicMock(bot=MagicMock(send_message=AsyncMock()))

    # Call the function
    await translate_and_send_messages(update, context, "Original message")

    # Check if the messages were sent in the expected languages
    expected_calls = [
        call(chat_id="chat1", text="Texte traduit en français", reply_to_message_id="msg1"),
        call(chat_id="chat1", text="Texto traducido al español", reply_to_message_id="msg1"),
    ]
    context.bot.send_message.assert_has_calls(expected_calls, any_order=True)

    # Check if translate_client.translate was called with the expected arguments
    expected_calls = [
        call("Original message", target_language="en"),
        call("Original message", target_language="fr"),
        call("Original message", target_language="es"),
    ]
    mock_translate_client.translate.assert_has_calls(expected_calls, any_order=True)



@pytest.fixture
def firestore_mock():
    with patch("helpers.firestore.Client") as mock_client:
        yield mock_client

@pytest.mark.asyncio
async def test_get_previous_messages():
    chat_id = "chat1"
    user_id = "user1"

    mock_doc_1 = MagicMock(id="1", to_dict=lambda: {"user_id": "user1", "message_text": "Test message 1", "role": "user", "timestamp": "2023-01-01"})
    mock_doc_2 = MagicMock(id="2", to_dict=lambda: {"user_id": "user1", "message_text": "Test response 1", "role": "assistant", "timestamp": "2023-01-02"})
    mock_doc_3 = MagicMock(id="2", to_dict=lambda: {"user_id": "user1", "message_text": "Test message 2", "role": "user", "timestamp": "2023-01-03"})

    with patch('helpers.db') as mock_db:
        mock_db.collection.return_value.document.return_value.collection.return_value.where.return_value.order_by.return_value.stream.return_value = [mock_doc_1, mock_doc_2, mock_doc_3]

        result = await helpers.get_previous_messages(chat_id, user_id)
        expected = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"},
            {"role": "user", "content": "Test message 2"}
        ]

        assert result == expected

@pytest.mark.asyncio
async def test_store_message_for_user():
    chat_id = "chat1"
    user_id = "user1"
    message_text = "Test message"

    with patch('helpers.db') as mock_db:
        await helpers.store_message(chat_id, user_id, message_text)
        mock_db.collection.return_value.document.return_value.collection.return_value.add.assert_called_once_with({
            'user_id': user_id,
            'message_text': message_text,
            'role': "user",
            'timestamp': firestore.SERVER_TIMESTAMP
        })

@pytest.mark.asyncio
async def test_store_message_for_assistant():
    chat_id = "chat1"
    user_id = "user1"
    message_text = "Test message"

    with patch('helpers.db') as mock_db:
        await helpers.store_message(chat_id, user_id, message_text, role='assistant')
        mock_db.collection.return_value.document.return_value.collection.return_value.add.assert_called_once_with({
            'user_id': user_id,
            'message_text': message_text,
            'role': "assistant",
            'timestamp': firestore.SERVER_TIMESTAMP
        })
