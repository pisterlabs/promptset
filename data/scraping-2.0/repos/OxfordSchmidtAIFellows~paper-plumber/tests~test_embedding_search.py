import os
from paperplumber.parsing.embedding_search import EmbeddingSearcher
from langchain.document_loaders import PyPDFium2Loader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


class TestEmbeddingSearcher:
    def setup_class(self):
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.pdf_path = os.path.join(current_directory, "maxwell2005.pdf")
        self.doc_embeddings = EmbeddingSearcher(self.pdf_path)

    def test_initialization(self):
        assert (
            self.doc_embeddings._pdf_path == self.pdf_path
        ), "PDF path is not initialized correctly"

    def test_pdf_exist(self):
        assert os.path.exists(self.doc_embeddings._pdf_path), "PDF file does not exist"

    def test_similarity_search(self):
        query = "Maxwell equations"
        results = self.doc_embeddings.similarity_search(question=query, k=2)

        # assuming results are list of strings
        assert isinstance(results, list), "Results should be a list"
        assert len(results) == 2, "Results length should match the k value"
