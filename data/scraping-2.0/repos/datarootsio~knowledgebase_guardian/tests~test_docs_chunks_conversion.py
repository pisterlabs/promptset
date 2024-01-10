from langchain.schema import Document
from pytest import LogCaptureFixture

from kb_guardian.docs_chunks_conversion import (
    create_document_chunks,
    load_text_documents,
    split_documents,
)


def test_load_text_documents() -> None:
    """Test whether some sample text documents are loaded correctly."""
    documents = load_text_documents("tests/files/Leuven.txt")

    for doc in documents:
        assert len(doc.page_content) > 0
        assert doc.metadata["source"] == "Leuven"


def test_split_documents() -> None:
    """Test whether documents are split correctly."""
    content = ".".join(["a" * 99 for _ in range(10)])
    meta = {"source": "test"}

    doc = Document(page_content=content, metadata=meta)

    splitted_docs = split_documents([doc], chunk_size=100, chunk_overlap=0)

    for doc in splitted_docs:
        assert 0 < len(doc.page_content) <= 100
        assert doc.metadata["source"] == "test"


def test_create_documents_chunks(caplog: LogCaptureFixture) -> None:
    """
    Test whether some sample files are correctly transformed into chunks.

    Also validate that a warning is raised for incorrect file encodings.
    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture and validate the logs.
    """
    splitted_docs = create_document_chunks("tests/files", "txt", 500, 0)

    assert len(splitted_docs) == 2
    assert "Document conversion failed for file 'Leuven_unreadable'" in caplog.text
