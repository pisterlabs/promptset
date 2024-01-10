import os
import logging
import datetime
from langchain.vectorstores import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from sqlalchemy import text
from app.utils import compute_file_hash, convert_pdf_to_text, get_punkt
from app.utils import split_text
import app.config as config


class PDFProcessor:
    """
    Processor for PDF files.
    """

    def __init__(self, **kwargs):
        embedding_model = HuggingFaceEmbeddings(
            model_name=kwargs.get(
                "embedding_model",
            )
        )

        self.vector_store = PGVector(
            connection_string=config.CONNECTION_STRING,
            embedding_function=embedding_model,
            collection_name="pdf_collection",
            collection_metadata={
                "collection_description": "Collection of PDF documents with embeddings",
                "collection_type": "pdf",
            },
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)
        get_punkt()

    def process_pdf(self, pdf_file):
        """
        Process a PDF file.

        :param pdf_file: The path to the PDF file.

        """
        try:
            self.logger.info("Processing %s", pdf_file)
            # Compute the hash of the file
            file_hash = compute_file_hash(pdf_file)

            if self.check_pdf(pdf_file):
                self.logger.info("PDF file %s already indexed", pdf_file)
                return

            # Convert the PDF to text
            pdf_texts = convert_pdf_to_text(pdf_file)
            # Split the pdf_texts into sentences
            sentences = split_text(pdf_texts)

            file_name: str = os.path.basename(pdf_file)[: -len(".pdf")]

            metadatas = [
                {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "timestamp": datetime.datetime.now().timestamp(),
                }
                for _ in sentences
            ]

            self.vector_store.add_texts(texts=sentences, metadatas=metadatas)

            self.logger.info("Finished processing %s", pdf_file)

        except Exception as err:
            self.logger.error("Error while processing %s: %s", pdf_file, err)

    def check_pdf(self, pdf_file):
        """
        Check if a PDF file has been indexed.

        :param pdf_file: The path to the PDF file.
        :return: True if the PDF file has been indexed, False otherwise.

        """
        file_hash = compute_file_hash(pdf_file)
        with self.vector_store.connect() as conn:
            if (
                conn.execute(
                    text(
                        f"SELECT cmetadata FROM langchain_pg_embedding WHERE cmetadata ->> 'file_hash' = '{file_hash}'"
                    )
                ).fetchone()
                is not None
            ):
                return True
        return False

    def search_str(self, query: str):
        """
        Search for a string in the PDF documents.

        :param query: The string to search for.
        """
        results = self.vector_store.similarity_search(query, 5)
        return results

    def search_str_without_md(self, query: str):
        """
        Search for a string in the PDF documents.

        :param query: The string to search for.
        """
        results = self.vector_store.similarity_search(query, 5)
        return [result.page_content for result in results]
