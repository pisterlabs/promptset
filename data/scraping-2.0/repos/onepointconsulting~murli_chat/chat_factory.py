from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from typing import Tuple, List
import os

from pathlib import Path

from log_factory import logger
from doc_loader import load_txt
from config import cfg


def extract_embeddings(texts: List[Document], doc_path: Path) -> FAISS:
    """
    Either saves the vector database embeddings locally or reads them from disk, in case they exist.
    :return a vector database wrapper around the embeddings.
    """
    embedding_dir = f"{cfg.faiss_persist_directory}/{doc_path.stem}"
    embedding_dir_path = Path(embedding_dir)
    if embedding_dir_path.exists() and len(list(embedding_dir_path.glob("*"))) > 0:
        return FAISS.load_local(embedding_dir, cfg.embeddings)
    # if Path(embedding_dir).exists():
    #     shutil.rmtree(embedding_dir, ignore_errors=True)
    try:
        docsearch = FAISS.from_documents(texts, cfg.embeddings)
        FAISS.from_texts
        docsearch.save_local(embedding_dir)
        logger.info("Vector database persisted")
    except Exception as e:
        logger.error(f"Failed to process {doc_path}: {str(e)}")
        if "docsearch" in vars() or "docsearch" in globals():
            docsearch.persist()
        return None
    return docsearch


def load_texts(doc_location: str) -> Tuple[List[str], Path]:
    """
    Loads the texts of the CSV file and concatenates all texts in a single list.
    :param doc_location: The document location.
    :return: a tuple with a list of strings and a path.
    """
    doc_path = Path(doc_location)
    texts = []
    failed_count = 0
    for i, p in enumerate(doc_path.glob("*.txt")):
        try:
            logger.info(f"Processed {p}")
            texts.extend(load_txt(p))
        except Exception as e:
            logger.error(f"Cannot process {p} due to {e}")
            failed_count += 1
    logger.info(f"Length of texts: {len(texts)}")
    logger.warning(f"Failed: {failed_count}")
    return texts, doc_path


def init_vector_search() -> FAISS:
    doc_location = os.environ["DOC_LOCATION"]
    logger.info(f"Using doc location {doc_location}.")
    doc_path = Path(doc_location)
    embedding_dir = f"{cfg.faiss_persist_directory}/{doc_path.stem}"
    embedding_dir_path = Path(embedding_dir)
    if embedding_dir_path.exists() and len(list(embedding_dir_path.glob("*"))) > 0:
        logger.info(f"reading from existing directory")
        docsearch = FAISS.load_local(embedding_dir, cfg.embeddings)
        return docsearch
    else:
        logger.warning(f"Cannot find path {embedding_dir} or path is empty.")
        logger.info("Generating vectors")
        texts, doc_path = load_texts(doc_location=doc_location)
        docsearch = extract_embeddings(texts=texts[: 6400 * 3], doc_path=Path(doc_path))
        return docsearch
