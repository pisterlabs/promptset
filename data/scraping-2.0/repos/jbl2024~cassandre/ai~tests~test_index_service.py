# pylint: disable=missing-docstring
import unittest
from unittest.mock import patch, MagicMock
from ai.services.index_service import load_and_split_pdf
from langchain.schema import Document as LangchainDocument


class TestLoadAndSplitPdf(unittest.TestCase):
    @patch("ai.services.index_service.PDFPlumberLoader")
    @patch("ai.services.index_service.SpacyTextSplitter")
    @patch("ai.services.index_service.parse_hints")
    def test_load_and_split_pdf_with_hints(
        self, mock_parse_hints, mock_SpacyTextSplitter, mock_PDFPlumberLoader
    ):
        # Set up the mock returns
        mock_PDFPlumberLoader.return_value.load.return_value = [
            LangchainDocument(page_content="sample page content", metadata={"page": 1})
        ]

        mock_text_splitter = MagicMock()
        mock_text_splitter.split_documents.return_value = [
            LangchainDocument(page_content="split page content", metadata={"page": 1})
        ]
        mock_SpacyTextSplitter.return_value = mock_text_splitter

        mock_document = MagicMock()
        mock_document.hints = "page:1: hint1; hint2"
        mock_parse_hints.return_value = {1: ["hint1", "hint2"], "all": []}

        # Call the function
        temp_file = MagicMock(name="tempfile.pdf")
        result_docs = load_and_split_pdf(temp_file, mock_document)

        # Assert the contents of the returned documents
        self.assertEqual(len(result_docs), 1)
        self.assertIn("split page content", result_docs[0].page_content)
        self.assertIn("hint1", result_docs[0].page_content)
        self.assertIn("hint2", result_docs[0].page_content)
