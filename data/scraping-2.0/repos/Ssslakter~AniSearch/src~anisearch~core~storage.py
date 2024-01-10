from venv import logger
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import qdrant_client.http.models as rest
from langchain.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from src.features import document_builder
from src.anisearch.configs import QdrantConfig

BATCH_SIZE = 64


class Storage:
    """Client wrapper to qdrant vector storage"""

    def __init__(self, embeddings: Embeddings, config: QdrantConfig) -> None:
        logger.info("Initializing storage")
        self.config = config
        self.embeddings = embeddings
        self.client = QdrantClient(url=config.url, api_key=config.api_key, timeout=100)
        self.document_builder = document_builder.DocumentBuilder()
        self.qdrant = Qdrant(
            client=self.client,
            collection_name=config.collection_name,
            embeddings=embeddings,
        )
        self.init_collection(config.collection_name)

    def init_collection(self, collection_name: str):
        """Tries to create collection, even if it already exists

        Raises:
            err_response: If qdrant returned error response not related with existance of collection
        """
        logger.info("Creating collection %s...", collection_name)
        vector_length = len(self.embeddings.embed_documents([""])[0])
        try:
            self.client.create_collection(
                collection_name,
                rest.VectorParams(
                    size=vector_length,
                    distance=rest.Distance[self.config.distance_func.upper()],
                ),
            )
            logger.info("Collection was created!")
        except UnexpectedResponse as err_response:
            if b"already exists!" in err_response.content:
                logger.info("Collection %s already exists", collection_name)
                return
            raise err_response

    def insert_df(self, df: pd.DataFrame):
        """Inserts documents to db from pandas dataframe"""
        logger.info("Starting to build documents...")
        df = self.document_builder.prepare_df(df)
        docs = self.document_builder.build_documents(df)
        logger.info("Documents were succesfully builded")
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : min(i + BATCH_SIZE, len(docs) - 1)]
            try:
                self.qdrant.add_documents(batch)
                logger.info(
                    "Uploaded %s/%s documents",
                    min(i + BATCH_SIZE, len(docs) - 1),
                    len(docs),
                )
            except UnexpectedResponse as e:
                logger.error(
                    "Failed to upload documents",
                    exc_info=e,
                )
                break
