from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from _common import _common as _common_


@_common_.exception_handler
def load_document(filepath: str) -> List:
    """
    Loads and processes a document from a given file path.

    This function reads a document from the specified file path and splits it into
    smaller chunks. Each chunk is a part of the document, divided based on character count,
    with a specified overlap between consecutive chunks.

    Args:
        filepath: The path to the file containing the document to be loaded.

    Returns:
        List: A list of document chunks, each represented as a string.


    """
    documents = TextLoader(filepath).load()
    return [doc.page_content for doc in
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80).split_documents(documents)]
