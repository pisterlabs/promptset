import logging
from typing import List
from docx import Document as DocxDocument
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.schema.document import Document


def process_docx(file_path: str) -> List[Document]:
    """
    Process a local DOCX file, extract text, and return the extracted text as a list of Document objects.

    Parameters:
    - file_path (str): The local path to the DOCX file to be processed.

    Returns:
    - List[Document]: A list of Document objects, where each object represents a page of the DOCX with its content.

    Raises:
    - Exception: Propagates any exceptions that occur during the processing of the DOCX.
    """
    try:
        logging.info(f"Processing DOCX file: {file_path}")

        # Process the DOCX
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()

        logging.info(f"DOCX processed successfully: {file_path}")

        # Return only 'documents' as a list
        return documents

    except Exception as e:
        logging.error(f"Error processing the DOCX: {e}")
        raise Exception(f"Error processing the DOCX: {e}")
