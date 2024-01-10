from unittest.mock import Mock
from langchain_functions.langchain_helper import get_response_from_query, get_qa_from_query, set_video_as_vector, get_response_qa_from_query_bolt

from langchain.schema.document import Document

def test_get_response_from_query():
    client = Mock()
    client.video_db.similarity_search.return_value = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
        Document(page_content="doc4")
    ]
    client.template = "template"
    client.openaiChat = Mock()
    # client.openaiChat.generate_response.return_value = {"response": "response"}
    client.video_meta = "video_meta"

    response, docs = get_response_from_query(client, "query")

    assert response is not None
    assert docs is not None

def test_get_qa_from_query():
    client = Mock()
    client.qa.return_value = {"answer": "answer"}
    client.chat_history = []

    answer = get_qa_from_query(client, "query")

    assert answer == "answer"
    assert client.chat_history == [("query", "answer")]

def test_set_video_as_vector():
    embeddings = Mock()
    embeddings.embed.return_value = "embedding"

    db, video_meta = set_video_as_vector("link", embeddings)

    assert db is not None
    assert video_meta is not None

def test_get_response_qa_from_query_bolt():
    app = Mock()
    app.openaiChat = "openaiChat"
    app.openaiQuestion = "openaiQuestion"
    app.document_db.as_retriever.return_value = "retriever"
    app.chat_history = []
    app.meta_data = "meta_data"

    answer, generated_question = get_response_qa_from_query_bolt("query", app, "chain_type")

    assert answer is not None
    assert generated_question is not None
    assert app.chat_history == [("query", answer)]
