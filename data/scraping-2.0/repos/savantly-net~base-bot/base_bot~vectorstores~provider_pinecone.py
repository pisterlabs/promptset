import logging
import os

import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, VectorStore

from base_bot.vectorstores.provider import VectorStoreProvider
from base_bot import config

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
default_variant = config.VECTORSTORE_DEFAULT_VARIANT

# check if params are set
if PINECONE_INDEX_NAME is None:
    raise ValueError("PINECONE_INDEX_NAME not set")
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY not set")
if PINECONE_ENV is None:
    raise ValueError("PINECONE_ENV not set")

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)

vectorstores = {
    "default": None,
}


class DefaultVectorStore(VectorStoreProvider):
    def __init__(self):
        # Initialize your VectorStore here
        pass

    def get_vectorstore(self, variant: str = default_variant):
        if variant == "":
            variant = "default"
        logging.info(f"getting vectorstore for variant {variant}")
        if vectorstores.get(variant) is None:
            logging.info("loading vectorstore...")
            vectorstores.update({variant: _load_vectorstore(variant)})
        return vectorstores.get(variant)


def _load_vectorstore(variant: str) -> VectorStore:
    logging.info("loading pinecone vectorstore...")
    pinecone.list_indexes()

    logging.info(f"variant: {variant}")
    logging.info("pinecone connection ok")

    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(
        PINECONE_INDEX_NAME, embeddings, namespace=variant
    )
