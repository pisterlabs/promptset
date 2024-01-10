from http import HTTPStatus
from io import BytesIO
from typing import List
from fastapi.testclient import TestClient

import os
from pathlib import Path
import pytest
import pytest_asyncio
from sqlmodel import SQLModel, Session, create_engine, select

from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app, get_session, get_current_user

# from app.core.auth import get_current_user
# from app.core.database import get_session
from app.models.document import Document, Chat, ChatMessage
from app.models.user import User

from .conftest import mock_get_current_user, mock_get_another_user


@pytest.mark.asyncio
async def test_list_documents(
    client: TestClient,
    documents: List[Document],
):
    app.dependency_overrides[get_current_user] = mock_get_current_user
    response = client.get("/documents")
    app.dependency_overrides.clear()
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()) == len(documents)
    assert all(doc["title"] in [d.title for d in documents] for doc in response.json())
    assert all("chats" in doc for doc in response.json())


# Now ensure that unauthenticated users get a 401
@pytest.mark.asyncio
async def test_list_documents_unauthenticated(client: TestClient):
    response = client.get("/documents")
    assert response.status_code == HTTPStatus.UNAUTHORIZED


# Now ensure that users only get their own documents
@pytest.mark.asyncio
async def test_list_documents_unauthorized(
    client: TestClient, documents: List[Document]
):
    app.dependency_overrides[get_current_user] = mock_get_another_user
    response = client.get("/documents")
    app.dependency_overrides.clear()
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()) == 0


@pytest.mark.asyncio
async def test_list_messages_in_chat(
    client: TestClient,
    documents: List[Document],
    chats: List[Chat],
):
    app.dependency_overrides[get_current_user] = mock_get_current_user
    response = client.get(f"/documents/{documents[0].id}/chat/{chats[2].id}")
    app.dependency_overrides.clear()
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_list_messages_in_chat_unauthenticated(
    client: TestClient,
    documents: List[Document],
    chats: List[Chat],
):
    response = client.get(f"/documents/{documents[0].id}/chat/{chats[2].id}")
    assert response.status_code == HTTPStatus.UNAUTHORIZED


@pytest.mark.asyncio
async def test_list_messages_in_chat_unauthorized(
    client: TestClient,
    documents: List[Document],
    chats: List[Chat],
):
    app.dependency_overrides[get_current_user] = mock_get_another_user
    response = client.get(f"/documents/{documents[0].id}/chat/{chats[2].id}")
    app.dependency_overrides.clear()
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()) == 0


# @pytest.mark.asyncio
# @patch.dict(os.environ, {"OPENAI_API_KEY": "mock_key"})  # Check that key isn't actually being used
# @patch("langchain.embeddings.openai.OpenAIEmbeddings", new_callable=MagicMock)
# async def test_upload_file(client: TestClient, session: Session):
#     app.dependency_overrides[get_current_user] = mock_get_current_user
#     docs_count = len(session.exec(select(Document.id)).all())
#     path_to_file = Path(__file__).parent / "sample.pdf"
#     with open(path_to_file, "rb") as f:
#         response = client.post(
#             "/documents/upload",
#             files={"uploaded_file": (str(path_to_file), f)},
#         )
#     new_docs_count = len(session.exec(select(Document)).all())
#     app.dependency_overrides.clear()
#     resp_json = response.json()
#     assert new_docs_count == docs_count + 1
#     assert response.status_code == HTTPStatus.OK
#     # FIXME: Should return document instead
#     assert isinstance(resp_json, int)
    
#     document = session.get(Document, resp_json)
#     assert document.title == "sample.pdf"
#     assert document.user_id == 1


# Test unauthorized upload
@pytest.mark.asyncio
async def test_upload_file_unauthorized(client: TestClient, session: Session):
    docs_count = len(session.exec(select(Document.id)).all())
    path_to_file = Path(__file__).parent / "sample.pdf"
    with open(path_to_file, "rb") as f:
        response = client.post(
            "/documents/upload",
            files={"uploaded_file": (str(path_to_file), f)},
        )
    new_docs_count = len(session.exec(select(Document)).all())
    assert new_docs_count == docs_count
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    
    
@pytest.mark.asyncio
async def test_create_new_chat(client: TestClient, session: Session, documents: List[Document]):
    document = documents[0]  # choose the first document
    
    # override get_current_user dependency
    app.dependency_overrides[get_current_user] = mock_get_current_user
    response = client.get(f"/documents/{document.id}/new_chat")
    app.dependency_overrides.clear()

    assert response.status_code == HTTPStatus.OK

    chat_id = response.json()
    # FIXME: Should return whole chat object
    assert isinstance(chat_id, int)

    # fetch the chat from the db to check it was created
    new_chat = session.get(Chat, chat_id)

    assert new_chat is not None
    assert new_chat.document_id == document.id


@pytest.mark.asyncio
async def test_create_new_chat_unauthenticated(client: TestClient, session: Session, documents: List[Document]):
    document = documents[0]  # choose the first document
    response = client.get(f"/documents/{document.id}/new_chat")
    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def mock_embed_message(message):
    return [0.9] * 1536

# Define a mock for get_ai_response function
async def mock_get_ai_response(chat_id, session, gpt_prompt, message):
    from langchain.schema import AIMessage
    return AIMessage(content="Mock AI response")

# Define a mock for the vector store retrieval function, since it can't be
# done with SQLite
def mock_get_similar_chunks(query_embedding, document_id, session, k=3):
    return [
        f"Mock similar chunk {n}"
        for n in range(0, k)
    ]

@pytest.mark.asyncio
@patch('app.api.document_router.embed_message', new=mock_embed_message)
@patch('app.api.document_router.get_ai_response', new=mock_get_ai_response)
@patch('app.api.document_router.get_k_similar_chunks', new=mock_get_similar_chunks)
@patch.dict(os.environ, {"OPENAI_API_KEY": "mock_key"})  # Check that key isn't actually being used
async def test_send_message(client: TestClient, session: Session, chats: List[Chat], user: User):
    app.dependency_overrides[get_current_user] = mock_get_current_user

    chat = chats[0]  # choose the first chat
    document_id = chat.document_id
    message = "Test message content"
    response = client.post(f"/documents/{document_id}/chat/{chat.id}/message", json={"message": message})
    app.dependency_overrides.clear()

    assert response.status_code == HTTPStatus.OK

    chat_message = response.json()
    assert chat_message['content'] == "Mock AI response"
    assert chat_message['chat_id'] == chat.id
    assert chat_message['user_id'] == user.id