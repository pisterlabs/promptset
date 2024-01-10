# app/tests/test_main.py
import json

from fastapi.testclient import TestClient
from langchain.schema import Document

from app.utils import save_uploaded_files, load_questions
from app.fastapi_app import app, process_question
from app.langchain_init import document_loader, text_splitter, openai_embeddings, llm, rag_prompt, format_docs

def test_save_uploaded_files(tmp_path):
    # Create temporary files
    document_file_path = tmp_path / "document.txt"
    questions_file_path = tmp_path / "questions.json"
    document_content = b"Sample document content"
    questions_content = b'{"questions": ["Sample Question 1", "Sample Question 2"]}'

    # Save files
    document_file_path.write_bytes(document_content)
    questions_file_path.write_bytes(questions_content)

    # Test the function
    saved_document_path, saved_questions_path = save_uploaded_files(document_file_path, questions_file_path)

    # Check if files were saved successfully
    assert saved_document_path.exists()
    assert saved_questions_path.exists()


def test_load_questions(tmp_path):
    # Create a temporary questions file
    questions_file_path = tmp_path / "questions.json"
    questions_content = b'{"questions": ["Sample Question 1", "Sample Question 2"]}'
    questions_file_path.write_bytes(questions_content)

    # Test the function
    loaded_questions = load_questions(questions_file_path)

    # Check if questions were loaded successfully
    assert loaded_questions is not None
    assert "questions" in loaded_questions
    assert len(loaded_questions["questions"]) == 2


class MockRetriever:
    pass


def test_process_question():
    # Create a sample question, document, and retriever
    question = {"q1": "Sample Question"}
    retriever = MockRetriever()

    # Test the function
    answer = process_question(question, retriever)

    assert answer is not None
    assert "question" in answer
    assert "answer" in answer


# Integration Test: Test the actual FastAPI application
def test_question_answering_endpoint(tmp_path):
    client = TestClient(app)

    # Create temporary files
    document_file_path = tmp_path / "document.txt"
    questions_file_path = tmp_path / "questions.json"
    document_content = b"Sample document content"
    questions_content = b'{"questions": ["Sample Question 1", "Sample Question 2"]}'

    # Save files
    document_file_path.write_bytes(document_content)
    questions_file_path.write_bytes(questions_content)

    # Make a request to the endpoint
    response = client.post("/qa/", files={"document_file": document_file_path, "questions_file": questions_file_path})

    # Check if the response is successful
    assert response.status_code == 200

    # Check if the response is in JSON format
    response_data = response.json()
    assert isinstance(response_data, list)

    # Check if each item in the response has the expected structure
    for item in response_data:
        assert "question" in item
        assert "answer" in item
