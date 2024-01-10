"""
This module defines functions for processing uploaded documents, including loading and chunking
content, as well as embedding text chunks and storing them in a vector collection.

It also provides utility functions to determine file formats and split content based on file
formats.
"""
import logging
import os
from typing import Optional

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from pydantic import HttpUrl

from .config import Settings
from .conversation import __run_summarise_chain
from .database import DocumentDAO
from .exceptions import UnsupportedFileFormatException
from .models import ProcStatus, DocumentWithMetadata, DocumentResponse
from .util import send_callback
from .vectorstore import generate_vectors_and_store

LOG = logging.getLogger(__name__)

settings = Settings.get_settings()
__FILE_FORMAT_DICT = {
    "md": "markdown",
    "txt": "text",
    "html": "html",
    "shtml": "html",
    "htm": "html",
    "pdf": "pdf",
}


def __get_file_format(file_path: str) -> Optional[str]:
    """
    Extracts the file format from a given file path.

    :param file_path: The path of the file.

    :returns Optional[str]: The detected file format or None if not supported.
    """
    file_path = os.path.basename(file_path)
    file_extension = file_path.split(".")[-1]
    return __FILE_FORMAT_DICT.get(file_extension, None)


def __load_and_split_content(file_path: str, file_format: str) -> list[Document]:
    """
    Loads and splits the content of a file based on the given file format.

    :param file_path: The path of the file to process.
    :param file_format: The format of the file.

    :returns list[DocumentResponse]: A list of DocumentResponse objects containing the split content.

    :raises UnsupportedFileFormatException: If the file format is not supported.
    """
    if file_path is None:
        raise FileNotFoundError(f"File path: {file_path} not found.")

    sentence_endings = [".", "!", "?", "\n\n"]
    words_breaks = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    # todo: support token text splitter and add file format based parameters
    if file_format == "html":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
            separators=RecursiveCharacterTextSplitter.get_separators_for_language(
                Language.HTML
            ),
        ).split_documents(UnstructuredHTMLLoader(file_path).load())

    if file_format == "markdown":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
            separators=RecursiveCharacterTextSplitter.get_separators_for_language(
                Language.MARKDOWN
            ),
        ).split_documents(UnstructuredMarkdownLoader(file_path).load())

    if file_format == "pdf":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
            separators=sentence_endings + words_breaks,
        ).split_documents(PyPDFLoader(file_path).load())

    if file_format == "text":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
            separators=sentence_endings + words_breaks,
        ).split_documents(TextLoader(file_path).load())

    raise UnsupportedFileFormatException(
        f"File: {file_path} with format {file_format} is not supported"
    )


async def process_document(
    document_dao: DocumentDAO,
    file_path: str,
    document_id: str,
    collection: str,
    callback_url: HttpUrl | None = None,
):
    """
    Processes an uploaded document. Runs the flow to chunk the file content and embed the text
    chunks and store it in a vector collection with other metadata

    :param document_dao: (DocumentDAO): Document data access object
    :param file_path: (str): Absolute path of the uploaded document to process
    :param document_id: (str): ID of the document
    :param collection: (str): CollectionRecord which the document to be added
    :param callback_url: (str)
    """
    LOG.debug("Processing document: %s", file_path)
    try:
        file_format = __get_file_format(file_path)
        chunks = __load_and_split_content(file_path, file_format)
        LOG.debug("File [%s] is split in to %d chunks", file_path, len(chunks))
        await generate_vectors_and_store(chunks, collection, document_id)

        updated_doc = await document_dao.update_document(
            DocumentWithMetadata(
                file_name=os.path.basename(file_path),
                process_status=ProcStatus.COMPLETE,
            )
        )

        if callback_url is not None:
            await send_callback(
                callback_url,
                DocumentResponse.model_validate(updated_doc),
            )

    except Exception as e:
        await document_dao.update_document(
            DocumentWithMetadata(
                file_name=os.path.basename(file_path),
                process_status=ProcStatus.ERROR,
                vectors=repr(e),
            )
        )
    LOG.debug("%s processing complete", file_path)


async def summarise(
    document: DocumentWithMetadata,
    update: bool = False,
    document_dao: DocumentDAO = None,
) -> DocumentWithMetadata:
    doc_list = [
        Document(page_content=e.document, metadata=e.cmetadata)
        for e in document.embeddings
    ]
    summary = await __run_summarise_chain(doc_list)
    document.summary = summary
    if update and document_dao is not None:
        document = await document_dao.update_document(document)
    return document
