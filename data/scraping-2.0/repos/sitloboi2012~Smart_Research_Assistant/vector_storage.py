from __future__ import annotations

from langchain.schema import Document as LangChainDocument
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.zilliz import Zilliz
from langchain.retrievers.zilliz import ZillizRetriever


from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    connections,
    utility,
    DataType,
)

from constant import EMBEDDING_FUNC

import streamlit as st


class ZillizVectorDatabase:
    """
    A class representing a Zilliz Vector Database.

    Attributes:
        cloud_uri (str): The URI of the cloud where the database is hosted.
        cloud_api_key (str): The API key for accessing the cloud.
        collection_name (str): The name of the collection in the database.
        embedding_function (OpenAIEmbeddings): The embedding function used for document embedding.

    Examples:
        >>> db = ZillizVectorDatabase()
        >>> db.insert_doc(LangChainDocument(page_content="test paper 5", metadata={"title": "this is a paper 5"}))
        >>> result = db.search_doc("paper 3")
    """

    def __init__(
        self,
        cloud_uri: str = st.secrets["ZILLIZ_CLOUD_URI"],
        cloud_api_key: str = st.secrets["ZILLIZ_API_KEY"],
        collection_name: str = st.secrets["ZILLIZ_COLLECTION_NAME"],
        embedding_function: OpenAIEmbeddings = EMBEDDING_FUNC,
    ):
        """
        Initializes a ZillizVectorDatabase object.

        Args:
            cloud_uri (str, optional): The URI of the cloud where the database is hosted. Defaults to "https://in03-b839a352e24af63.api.gcp-us-west1.zillizcloud.com".
            cloud_api_key (str, optional): The API key for accessing the cloud. Defaults to "ac7fdae203ef6b731136e454178d946f87353b422d3b5f40e4f573bf62d6b4c483da92b8e26776aa63008f1a35469b9ac50a5785".
            collection_name (str, optional): The name of the collection in the database. Defaults to "Production".
            embedding_function (OpenAIEmbeddings, optional): The embedding function used for document embedding. Defaults to EMBEDDING_FUNC.
        """
        self.embedding_function = embedding_function
        self.cloud_uri = cloud_uri
        self.cloud_api_key = cloud_api_key

        # DATABASE CONNECTION
        connections.connect("default", uri=cloud_uri, token=cloud_api_key)

        # DATABASE SCHEMA
        if utility.has_collection(collection_name) is not True:
            paper_id = FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                description="unique id of the document",
            )
            embed_field = FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=1536,
                description="vector embedding of the document",
            )
            document = FieldSchema(
                name="document", dtype=DataType.JSON, description="text of the document"
            )
            metadata = FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="metadata of the document",
            )
            schema = CollectionSchema(
                fields=[paper_id, embed_field, document, metadata],
                auto_id=False,
                description="Production Collection",
            )
            self.collection = Collection(
                name=collection_name,
                schema=schema,
                using="default",
                enable_dynamic_field=True,
                auto_id = True
            )
            self.collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                },
            )
        else:
            self.collection = Collection(name=collection_name, enable_dynamic_field=True)

    def embed_documents(self, document: LangChainDocument) -> list[float]:
        """
        Embeds a list of documents using the embedding function.

        Args:
            document (LangChainDocument): The document to be embedded.

        Returns:
            list[float]: The embedded representation of the document.
        """
        return self.embedding_function.embed_documents([document])

    def insert_doc(self, document: LangChainDocument) -> None:
        """
        Inserts a document into the collection.

        Args:
            document (LangChainDocument): The document to be inserted.

        Returns:
            None
        """
        embed_text = self.embed_documents(document.page_content)
        db_length = self.collection.num_entities
        self.collection.insert(
            [
                [db_length + 1],
                embed_text,
                [{"raw_text": document.page_content}],
                [{"metadata": document.metadata}],
            ]
        )
        self.collection.flush()
        print("Document inserted successfully.")

    def search_doc(self, query: str) -> list[dict[str, list | dict]]:
        """
        Searches for a document in the collection based on the given query.

        Args:
            query (str): The query string to search for.

        Returns:
            list[dict[str, list | dict]]: A list of dictionaries containing the search results.
        """
        embedded_query = self.embedding_function.embed_query(query)
        result = self.collection.search(
            [embedded_query],
            anns_field="embed_vector",
            param={"metric_type": "L2"},
            limit=2,
            output_fields=["document", "metadata"],
        )

        search_results = []
        for hits in result:
            hits_dict = {
                "ids": hits.ids,
                "distances": hits.distances,
                "documents": [hit.entity.get("document") for hit in hits],
                "metadata": [hit.entity.get("metadata") for hit in hits],
            }
            search_results.append(hits_dict)

        return search_results


# test
#db = ZillizVectorDatabase()
# db.insert_doc(
#    LangChainDocument(
#        page_content="test paper 5", metadata={"title": "this is a paper 5"}
#    )
# )
#result = db.search_doc("paper 3")
#print(result)
#ZillizRetriever(
#    embedding_function= db.embedding_function,
#    collection_name = "Production",
#    connection_args = {
#        "uri": db.cloud_uri,
#        "token": db.cloud_api_key,
#        "secure": True,
#    },
#    consistency_level = "Session",
#    search_params = {
#        "metric_type": "L2",
#        "params": {"nprobe": 10},
#    },
#)


db = ZillizVectorDatabase()
langchain_db = Zilliz(
    embedding_function = EMBEDDING_FUNC,
    collection_name = "Production",
    connection_args = {
        "uri": db.cloud_uri,
        "token": db.cloud_api_key,
        "secure": True,
    },
    consistency_level = "Session",
    primary_field="pk",
    text_field="document",
    vector_field="vector"
)

ziliz_retriever = ZillizRetriever(
    embedding_function = EMBEDDING_FUNC,
    collection_name = "Production",
    connection_args = {
        "uri": db.cloud_uri,
        "token": db.cloud_api_key,
        "secure": True,
    },
    consistency_level = "Session",
    primary_field="pk",
    text_field="document",
    vector_field="vector",
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    #search_params={"search_type": "mmr", "param": {"lambda": 0.5, "k": 10}}
)