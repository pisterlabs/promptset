import logging
import os
import shutil

from jsonargparse import CLI
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from autodev.qa.embedding import CachedOpenAIEmbeddings
from autodev.qa.indexing import FileInfo, load_lc_documents
from autodev.qa.splitting import PythonAstSplitter

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

log = logging.getLogger("index_source_code")

extension_to_splitter = {
    ".py": PythonAstSplitter(),
    ".md": MarkdownTextSplitter(),
}
default_splitter = RecursiveCharacterTextSplitter()


def splitter_factory(file_info: FileInfo):
    if not file_info.extension:
        return default_splitter
    return extension_to_splitter.get(file_info.extension, default_splitter)


def get_embeddings_generator(embeddings_cache_file: str):
    embeddings = CachedOpenAIEmbeddings()
    if embeddings_cache_file and os.path.exists(embeddings_cache_file):
        embeddings.load_cache(embeddings_cache_file)
    return embeddings


def get_vectorstore(embeddings: Embeddings, db_directory: str):
    return Chroma(persist_directory=str(db_directory), embedding_function=embeddings)


def generate_document_id(doc: Document):
    metadata = doc.metadata
    try:
        path = metadata["source"]
        last_modified = metadata["last_modified"]
        part = metadata["part"]
        num_parts = metadata["num_parts"]
        size = metadata["size"]
    except KeyError as e:
        raise KeyError(f"Missing metadata in document: {doc}: {e}")
    return f"{path=}:{last_modified=}:{part}/{num_parts}:{size=}"


def index_codebase(
    dir: str = "src",
    db: str = "chroma_db",
    cache: str = ".openai_embeddings_cache",
    reset: bool = True,
):
    """
    Index a codebase, create embeddings and store them in a vectorstore.

    Args:
        dir: Directory to index.
        db: Vectorstore database directory.
        cache: Embeddings cache file.
        reset: Delete the vectorstore directory before indexing.
    """
    if reset and os.path.exists(db):
        shutil.rmtree(db)
        log.info(f"Deleted existing vectorstore directory: {db}")

    documents = list(load_lc_documents(dir, splitter_factory=splitter_factory))
    doc_ids = [generate_document_id(doc) for doc in documents]
    embeddings = get_embeddings_generator(cache)
    vectorstore = get_vectorstore(embeddings, db)
    try:
        vectorstore.add_documents(documents=documents, doc_ids=doc_ids)
    except Exception as e:
        log.error(f"Error adding documents to vectorstore due to {e}")
        raise
    finally:
        embeddings.save_cache(cache)
        log.info(f"Saved embeddings cache to {cache}")
        vectorstore.persist()
        log.info(f"Persisted vectorstore to {db}")

    log.info("Finished indexing codebase")


if __name__ == "__main__":
    CLI(index_codebase)
