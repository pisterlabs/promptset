import logging
from typing import Any, Callable

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.pgvector import PGVector
from pydantic import BaseModel

from ..config import settings
from .directory_utils import initialize_dir_for_file
from .embedding_selector import embedding_selector
from .indexer_utils.postgres_methods import make_connection_string


class VecDBMethods(BaseModel):
    upsert: Callable | str
    save: Callable | str | None


def get_vecdb_method(vector_db_name: str) -> VecDBMethods:
    """
    Sets the vector database method based on the vector database name.

    Parameters:
        vector_db_name (str): The name of the vector database to be used.

    Returns:
        VecDBMethods: contains the callables to database methods in the object attributes
    """
    dict_db_method = {
        "FAISS": {"upsert": FAISS.from_documents, "save": "save_local"},
        "postgres": {
            "upsert": PGVector.from_documents,
        },
    }
    # vecdb_methods = VecDBMethods(upsert=dict_db_method[vector_db_name]['upsert'], save=dict_db_method[vector_db_name]['save'])
    vecdb_methods = VecDBMethods(**dict_db_method[vector_db_name])
    return vecdb_methods


def get_extra_arguments() -> dict:
    """
    get extra arguments for the upsert method based on the database
    """
    arg_dict = {}
    if settings.database_type == "postgres":
        arg_dict["connection_string"] = make_connection_string()
        arg_dict["collection_name"] = settings.database_table
        arg_dict["pre_delete_collection"] = settings.indexer_override
    elif settings.database_type == "FAISS":
        pass  # no extra arguments for FAISS db

    return arg_dict


def split_document(doc_path: str, chunk_size: int, overlap: int) -> Any:
    """
    Splits a document into chunks. The chunks are split along using these separators:
    "\n\n", "\n", " ", ""
    starting from the first, trying to use it to meet a certain chunk-size requirement,
    and moving to the following if that is not possible (trying to keep paragraphs, sentences
    and finally words intact).

    Parameters:
        doc_path (str): The path to the document to be split.
        chunk_size (int): size of each chunk, measured in number of characters
        overlap (int): amount of overlap between different chunks, measured in number of characters

    Returns:
        Any: The split documents.
    """
    raw_document = TextLoader(doc_path).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    documents = text_splitter.split_documents(raw_document)
    return documents


def index_from_document(embeddings_function: Callable, upsert_method: Callable) -> Any:
    """
    Makes an index of the document.
    """
    docs = split_document(
        settings.document_path, chunk_size=settings.chunk_size, overlap=settings.overlap
    )
    logging.info(
        f"Created {len(docs)} vectors, using chunk_size={settings.chunk_size} and overlap={settings.overlap}"
    )
    args_dict = get_extra_arguments()
    index = upsert_method(documents=docs, embedding=embeddings_function(), **args_dict)
    return index


def write_database_to_file(
    index: Any, path: str, save_method: Callable
) -> (
    None
):  # TODO this at least works for FAISS, need to see if there are similar methods for other dbs
    """
    Writes the database to a file.
    """
    initialize_dir_for_file(path)
    save_function = getattr(index, save_method)
    save_function(path)


def indexer(save_to_file: bool = False) -> None:
    # get the methods associated with the selected database type
    vecdb_methods = get_vecdb_method(settings.database_type)
    # get the embeddings function by name
    embeddings = embedding_selector(settings.embeddings_name)
    # create a vector index starting from a document
    index = index_from_document(
        embeddings_function=embeddings, upsert_method=vecdb_methods.upsert
    )
    if save_to_file:
        # save the vector index to file
        write_database_to_file(
            index=index, path=settings.database_path, save_method=vecdb_methods.save
        )


if __name__ == "__main__":
    # create a new vector index in a database
    indexer()
