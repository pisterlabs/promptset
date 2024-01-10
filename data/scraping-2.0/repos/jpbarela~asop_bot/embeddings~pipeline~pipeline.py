from typing import List

from langchain.document_loaders import PyPDFLoader
import os
from pathlib import Path
from vectorstore import save_documents, delete_store


def process():
    """
        Loads files from the downloaded_pdfs directory into a vector database.

        Will delete the existing data store
    """
    delete_store()
    pdf_names = _get_pds_from_directory()

    for pdf in pdf_names:
        loader = PyPDFLoader(
            str(_get_downloaded_pdf_directory().joinpath(pdf)))
        pages = loader.load_and_split()
        save_documents(pages)


def _get_pds_from_directory() -> List[str]:
    """
        Retrieves a list of the files in download_pdfs directory
    """
    return [file for file in os.listdir(_get_downloaded_pdf_directory()) if file[-4:] == ".pdf"]


def _get_downloaded_pdf_directory() -> Path:
    return Path(__file__).parent.parent.joinpath("downloaded_pdfs")
