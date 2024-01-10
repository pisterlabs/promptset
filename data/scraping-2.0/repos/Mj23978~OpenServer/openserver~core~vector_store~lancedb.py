from langchain.vectorstores.lancedb import LanceDB
import lancedb

from .embedding.base import BaseEmbedding
from .base import VectorStore


class LanceDBVectorStore(VectorStore):

    def __init__(self, index_name: str, db_path: str, embeddings: BaseEmbedding, api_key: str | None = None):
        self.db = lancedb.connect(db_path, api_key=api_key)
        if index_name not in self.db.table_names():
            self.table = self.db.create_table(
                name=index_name,
                data=[{
                    "id": "1",
                    "text": "Hello World",
                    "vector": embeddings.client.embed_query("Hello World"),
                }],
                mode="overwrite",
            )
        else:
            self.table = self.db.open_table(index_name)
        self.embeddings = embeddings
        self.client = LanceDB(connection=self.table,
                              embedding=embeddings.client)
