import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import pandas as pd
from pymilvus.exceptions import (
    ConnectionNotExistException,
)
from pymilvus import Collection, utility

# langchain
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from exceptions.exceptions import DatabaseException


class Vecstore:
    """
    vector store utility class for milvus database
    """

    def __init__(self, host: str = "localhost", port: str = "19530"):
        """configuring a vector store database connection and setting static variables

        Args:
            host (str): host of database
            port (str): port of database

        Raises:
            ConnectionNotExistException: if the connection is not successful
        """
        self.DEFAULT_INDEX_PARAMS = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.INDEX_DATA = {
            "field_name": "embeddings",
            "index_params": self.DEFAULT_INDEX_PARAMS,
        }
        self.SEARCH_PARAMS = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
            "offset": 5,
        }
        self.host = str(host)
        self.port = str(port)
        try:
            connections.connect(host=self.host, port=self.port)
        except ConnectionNotExistException:
            raise ConnectionNotExistException(
                f"Check your database connection with {self.host +':'+ self.port}"
            )

        id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            description="primary key",
            is_primary=True,
            auto_id=True,
        )
        embeddings = FieldSchema(
            name="embeddings",
            dtype=DataType.FLOAT_VECTOR,
            dim=1536,  # max limit 32,768
            description="Embeddings",
        )
        content = FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=30000,
            description="text content",
        )

        self.schema = CollectionSchema(fields=[id, embeddings, content])

        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    def setup_new_collection(self, collection_name: str, collection_data: str):
        """setup new collection with index and data entry
        Args:
            collection_name (str): uploaded file name
            collection_data (str): uploaded file data
        """
        self.create_collection(collection_name)
        try:
            self.insert_file_data(collection_name, collection_data)
            self.create_index(collection_name)
        except BaseException as exception:
            utility.drop_collection(collection_name)
            raise DatabaseException(f"Problem with: {exception}")

    def create_collection(
        self,
        collection_name: str,
    ):
        """creating collection with default database

        Args:
            collection_name (str): name of the collection
        """
        Collection(
            name=collection_name, schema=self.schema, using="default", shards_num=2
        )

    def release_all(self):
        """
        Releasing all the collection loaded into the memory
        """
        all_collections = utility.list_collections()
        for collection in all_collections:
            Collection(collection).release

    def load_collection(self, collection_name: str):
        """
        loading a collection into the memory
        Args:
            collection_name (str): name
        """
        if collection_name in utility.list_collections():
            Collection(collection_name).load()
            print("loaded")

    def list_all_collections(self):
        """
        listing all the collection with configured indexes by default

        Returns:
            list: name of indexed collection
        """
        all_collection = utility.list_collections()
        indexed_collection = []
        for collection in all_collection:
            if Collection(collection).indexes:
                indexed_collection.append(collection)
        return sorted(indexed_collection)

    def create_index(self, collection_name):
        """creating index for given collection's embeddings

        Args:
            collection_name (str): name of collection
        """
        collection = Collection(collection_name)
        collection.create_index(
            field_name="embeddings",
            index_params=self.DEFAULT_INDEX_PARAMS,
            index_name=collection.name + "_embeddings_index",
        )
        utility.index_building_progress(collection_name)

    def insert_file_data(self, collection_name: str, file_data):
        """
        managing chunking and embeddings creation of file data
        and pushing data to collection dataset

        Args:
            collection_name (str): name
            file_data (str): data file
        """
        collection = Collection(collection_name)
        data_chunks = self.create_vecstore_data_chunks(file_data)
        print(len(data_chunks))
        for i, chunk in enumerate(data_chunks):
            embeddings = self.create_vecstore_embeddings(chunk)
            chunk = [chunk]
            data = [embeddings, chunk]
            insert_status = collection.insert(data)
            print(insert_status, collection_name, i)

    def search_with_index(self, collection_name: str, query: str):
        """
        searching collection based on given given query
        (creating embedding of the query and stating a similarity search with
        Euclidean distance in vector store)

        Args:
            collection_name (str): name
            query (str): question asked

        Returns:
            _type_: _description_
        """
        collection = Collection(collection_name)
        embed_query = self.create_vecstore_embeddings(query, query=True)
        results = collection.search(
            data=[embed_query],
            anns_field="embeddings",
            param=self.SEARCH_PARAMS,
            limit=3,
            expr=None,
            output_fields=["content"],
            consistency_level="Strong",
        )

        return results

    def create_vecstore_data_chunks(self, text: str) -> list:
        """splitting large data into multiple chunks

        Args:
            text (str): document text

        Returns:
            list: of string
        """
        text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=60)
        split_texts = text_splitter.split_text(text)
        print("\n\nCreated Data Chunks...")
        return split_texts

    def create_vecstore_embeddings(self, text: list, query: bool = False) -> list:
        """
        querying openai api for creating embeddings

        Args:
            text (list): given query text
            query (bool, optional): functional change for a query. Defaults to False.

        Returns:
            list: _description_
        """
        embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        if query:
            embeddings = embeddings_model.embed_query(text)
            return embeddings
        embeddings = embeddings_model.embed_documents([text])
        print("\nCrated the embeddings ........")
        return embeddings
