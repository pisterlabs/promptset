# from: https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pgvector.html
import sys
from io import StringIO
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from more_itertools import chunked


# from: https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/markdown.html
from langchain.document_loaders import UnstructuredFileIOLoader
import sqlalchemy

from unstructured.__version__ import __version__ as __unstructured_version__  # type: ignore
from unstructured.partition.md import partition_md  # type: ignore
from splitgraph_chatgpt_plugin.config import DOCUMENT_COLLECTION_NAME

from .persistence import connect, get_embedding_store_pgvector
from .ddn import get_repo_list, RepositoryInfo
from .markdown import repository_info_to_markdown
from .config import get_db_connection_string, get_openai_api_key
from contextlib import closing

EMBEDDING_CHUNK_SIZE = 50
DOCUMENT_CHUNK_BYTES = 1000

DELETE_OLD_EMBEDDINGS_QUERY = """
    DELETE FROM langchain_pg_embedding
    WHERE
        collection_id = (select uuid from langchain_pg_collection where name = :collection) AND
        cmetadata->>'namespace' = :namespace AND
        cmetadata->>'repository' = :repository;
    """


# Based on UnstructuredMarkdownLoader
class UnstructuredMarkdownIOLoader(UnstructuredFileIOLoader):
    """Loader that uses unstructured to load markdown files."""

    def _get_elements(self) -> List:
        _unstructured_version = __unstructured_version__.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )

        return partition_md(file=self.file, **self.unstructured_kwargs)


def remove_old_embeddings(
    connection: sqlalchemy.engine.Connection, collection: str, docs: List[Document]
) -> None:
    for doc in docs:
        stmt = sqlalchemy.text(DELETE_OLD_EMBEDDINGS_QUERY)
        stmt = stmt.bindparams(
            collection=collection,
            namespace=doc.metadata["namespace"],
            repository=doc.metadata["repository"],
        )
        connection.execute(stmt)
    connection.commit()


def prepare_repository_info_documents(
    repository_info: RepositoryInfo,
) -> List[Document]:
    repository_info_markdown = repository_info_to_markdown(repository_info)
    loader = UnstructuredMarkdownIOLoader(StringIO(repository_info_markdown))
    documents = loader.load()
    # add metadata to documents
    for d in documents:
        d.metadata["namespace"] = repository_info.namespace
        d.metadata["repository"] = repository_info.repository
    text_splitter = CharacterTextSplitter(
        chunk_size=DOCUMENT_CHUNK_BYTES, chunk_overlap=0
    )
    return text_splitter.split_documents(documents)


def main() -> None:
    # set repo_index_limit to an integer to only index the first N repos.
    repo_index_limit = None
    collection = DOCUMENT_COLLECTION_NAME
    namespace = sys.argv[1]
    with closing(connect(get_db_connection_string())) as connection:
        vstore = get_embedding_store_pgvector(
            connection, collection, get_openai_api_key()
        )
        print(f"Indexing repositories in namespace {namespace}")
        repo_list = get_repo_list(namespace)
        repository_info_documents: List[Document] = []
        for repo_info in repo_list[0:repo_index_limit]:
            repository_info_documents.extend(
                prepare_repository_info_documents(repo_info)
            )
        print(
            f"Calculating embeddings for {len(repository_info_documents)} documents, {EMBEDDING_CHUNK_SIZE} at a time"
        )
        for chunk in chunked(repository_info_documents, EMBEDDING_CHUNK_SIZE):
            remove_old_embeddings(connection, collection, chunk)
            vstore.add_documents(chunk)


main()
