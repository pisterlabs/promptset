import logging
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema.document import Document
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_pdf(file_path: str) -> List[Document]:
    """
    Processes a PDF file from a local file path.

    Parameters:
    - file_path (str): The path to the PDF file.

    Returns:
    - List[Document]: A list of Document objects.
    """
    try:
        logging.info(f"Starting PDF processing for file: {file_path}")

        # Load and process the PDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        number_of_pages = len(documents)
        logging.info(f"Number of pages processed: {number_of_pages}")

        logging.info("PDF processing completed successfully")
        return documents

    except Exception as e:
        logging.error(f"Error processing the PDF: {e}")
        raise Exception(f"Error processing the PDF: {e}")
