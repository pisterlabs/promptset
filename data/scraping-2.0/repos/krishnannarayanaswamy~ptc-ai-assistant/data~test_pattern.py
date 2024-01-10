from cassandra.cluster import Session
from cassandra.query import BatchStatement
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.cassandra import Cassandra
from abc import ABC, abstractmethod
from typing import List
from model import Product
import json


class TestPattern(ABC):
    """
    Abstract class for test pattern
    """

    def __init__(self, session: Session, keyspace: str):
        self.session = session


    def search(self, query: str, k=100) -> List[Product]:
        """
        Query
        """
        result = self.vectore_store().similarity_search(query, k)
        products: List[Product] = []
        for prod in result:
            metadata = prod.metadata
            product = Product(item_code=metadata['item_code'],
                              item_name=metadata['item_name'],
                              description=metadata['description'],
                              price=metadata['price'],
                              availability=metadata['availability'])
            products.append(product)

        # Insert results to result store
        batch = BatchStatement()
        for i, prod in enumerate(products):
            batch.add(self.result_store_insert,
                      (query, self.name(), i + 1, prod.product_id, False))
        try:
            self.session.execute(batch)
        except Exception as e:
            print("Storing result failed", e)

        return products

    def get_previous_results(self) -> List[Product]:
        pass

    @abstractmethod
    def name(self) -> str:
        """
        This name will be used as an identifier for the test pattern
        """
        pass

    @abstractmethod
    def vectore_store(self) -> VectorStore:
        pass

    @abstractmethod
    def embeddings(self) -> Embeddings:
        pass


class OpenAITestPattern(TestPattern):
    """
    Test pattern for OpenAI embedding model `text-embedding-ada-002`.
    For text embeddings, this pattern constructs a text in the following format:
    ```
    Item Code: {['item_code']}
    Item Name: {['item_name']}
    Description: {['description']} 
    Available: {['availability']} 
    Price: {['price']}

    ```
    """

    def __init__(self, session: Session, model_name: str, api_key: str, keyspace: str, table_name: str):
        super().__init__(session=session, keyspace=keyspace)

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=self.model_name,
                                           openai_api_key=api_key)
        self._vstore = Cassandra(embedding=self.embeddings,
                                 session=session,
                                 keyspace=keyspace,
                                 table_name=table_name)

    def vectore_store(self) -> VectorStore:
        return self._vstore

    def name(self) -> str:
        return f"openai_{self.model_name}_v1"

    def embeddings(self) -> Embeddings:
        return self.embeddings()