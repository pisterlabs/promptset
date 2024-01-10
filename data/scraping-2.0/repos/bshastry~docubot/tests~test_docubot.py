import unittest
from docubot import build_kb
from text_utils.text_utils import tiktoken_len
from langchain.docstore.document import Document


class TestDocuBot(unittest.TestCase):
    """
    A class for testing the functionality of the DocuBot application.
    """

    def test_build_kb(self):
        chunks = build_kb("test_files")
        # Test if the function returns a list
        self.assertIsInstance(chunks, list)

        # Test if a chunk is a document
        self.assertIsInstance(chunks[0], Document)

        # Test if the function returns a list of chunks with at most 512 tokens per chunk
        for c in chunks:
            self.assertLessEqual(tiktoken_len(c.page_content), 512)


if __name__ == "__main__":
    unittest.main()
