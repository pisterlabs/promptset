import os
import unittest

from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from src.main.app import get_pdf_text, get_text_chunks, get_vector_store


class TestLangChainLab(unittest.TestCase):
    def setUp(self) -> None:
        load_dotenv()

    def test_get_pdf_text(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, "..", "..")
        folder_path = os.path.join(root_dir, "resources")
        test_pdf_path = os.path.join(folder_path, "langchain.pdf")
        text = get_pdf_text(test_pdf_path)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_get_pdf_text_with_invalid_path(self):
        invalid_pdf_path = "non_existent_path/non_existent_file.pdf"
        with self.assertRaises(FileNotFoundError):
            get_pdf_text(invalid_pdf_path)

    def test_get_text_chunks(self):
        sample_text = "This is a test. " * 100
        chunks = get_text_chunks(sample_text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertTrue(len(chunks) > 0)

    def test_get_vector_store(self):
        sample_chunks = ["This is a test.", "Another test sentence."]
        vector_store = get_vector_store(sample_chunks)
        self.assertIsInstance(vector_store, FAISS)


if __name__ == "__main__":
    unittest.main()
