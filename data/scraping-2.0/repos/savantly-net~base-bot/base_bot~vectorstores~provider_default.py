import logging
import pickle
from pathlib import Path
from typing import Optional

from langchain.vectorstores import VectorStore

from base_bot.vectorstores.provider import VectorStoreProvider

from base_bot import config
from base_bot.document_loader import get_document_loader
from base_bot.ingest import ingest_docs

vectorstore: Optional[VectorStore] = None
vectorstore_variants = config.VECTORSTORE_VARIANTS
default_variant = config.VECTORSTORE_DEFAULT_VARIANT

class DefaultVectorStore(VectorStoreProvider):
    def __init__(self):
        # Initialize your VectorStore here
        pass

    def get_vectorstore(self, variant: str = default_variant):
        global vectorstore
        if vectorstore is None:
            vectorstore = _load_vectorstore(variant)
        return vectorstore


def _load_vectorstore(variant: str) -> VectorStore:
    logging.info("loading pickled vectorstore...")
    logging.info(f"variant: {variant if variant else 'default'}")

    vectorstore_path = config.VECTORSTORE_PATH
    
    if variant and variant in vectorstore_variants:
        vectorstore_path = f"{config.VECTORSTORE_PATH}.{variant}"

    if not Path(vectorstore_path).exists():
        if config.VECTORSTORE_CREATE_IF_MISSING:
            logging.info(f"{vectorstore_path} does not exist, creating from docs")

            ingest_docs(
                document_loader=get_document_loader(config.DOCS_PATH),
                vectorstore_path=vectorstore_path,
            )
        else:
            raise ValueError(
                f"{vectorstore_path} does not exist, please run ingest.py first"
            )
    with open(vectorstore_path, "rb") as f:
        vstore = pickle.load(f)
        logging.info("loaded vectorstore")
        return vstore
