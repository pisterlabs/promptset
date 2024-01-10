import pytest
from unittest.mock import Mock, patch
from api.app.services.pdf_processing import process_pdf
from langchain.schema.document import Document
from typing import List


@patch('api.app.services.pdf_processing.PyMuPDFLoader')
def test_process_pdf_success(mock_loader: Mock) -> None:
    """
    Test to ensure that the process_pdf function correctly processes a simulated PDF.

    This test simulates the behavior of PyMuPDFLoader returning mock documents,
    and then checks whether the process_pdf function processes these documents as expected.

    Args:
        mock_loader (Mock): A mock object of PyMuPDFLoader.
    """
    # Set up simulated behavior
    mock_docs: List[Document] = [
        Document(page_content="Test Content 1", metadata={"page": 1}),
        Document(page_content="Test Content 2", metadata={"page": 2})
    ]
    mock_loader.return_value.load.return_value = mock_docs

    # Execute the function with a simulated file path
    documents = process_pdf("fake_path.pdf")

    # Perform assertions
    assert len(documents) == 2
    assert documents[0].page_content == "Test Content 1"
    assert documents[1].page_content == "Test Content 2"
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2


@patch('api.app.services.pdf_processing.PyMuPDFLoader')
def test_process_pdf_error(mock_loader: Mock) -> None:
    """
    Test the error handling in the process_pdf function.

    This test simulates a situation where PyMuPDFLoader throws an exception,
    and then checks whether the process_pdf function correctly propagates this exception.

    Args:
        mock_loader (Mock): A mock object of PyMuPDFLoader.
    """
    # Set up the mock to throw an exception
    mock_loader.return_value.load.side_effect = Exception("Error de carga")

    # Verify that an exception is thrown as expected
    with pytest.raises(Exception) as exc_info:
        process_pdf("fake_path.pdf")
    assert "Error de carga" in str(exc_info.value)
