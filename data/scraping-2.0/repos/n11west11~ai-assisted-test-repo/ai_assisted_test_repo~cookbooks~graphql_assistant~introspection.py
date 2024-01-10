from typing import Any, Dict

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import build_client_schema, get_introspection_query, print_schema
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from ai_assisted_test_repo.tools.embeddings import cached_embedder


class GraphQLBaseParameters(BaseModel):
    headers: Dict[str, Any] = Field(
        {}, description="The headers to use in the GraphQL server"
    )
    url: str = Field(..., description="The GraphQL server url")


def introspect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(url, headers=headers)
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = graphql_client.execute(gql(introspection))
        schema_str = print_schema(build_client_schema(result))
        return schema_str
    except Exception as e:
        return "Error: " + str(e)


async def aintrospect(url: str, headers=None):
    """Introspect a GraphQL server to fetch high level information about the schema"""
    if headers is None:
        headers = {}
    try:
        transport = AIOHTTPTransport(url, headers=headers)
        graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
        introspection = get_introspection_query()
        result = await graphql_client.execute_async(gql(introspection))
        schema_str = print_schema(build_client_schema(result))
        return schema_str
    except Exception as e:
        return "Error: " + str(e)


def get_introspection_texts(query: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents


async def aget_introspection_texts(query: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([query])
    return documents


def introspection_db(endpoint) -> VectorStore:
    """
    This function is used to produce a vector store that
    will be used as context for the graphql query
    :param endpoint: The endpoint that will be used to produce the vector store
    :return: A vector store that will be used as context for the graphql query
    """
    introspection_result = introspect(endpoint)
    introspection_texts = get_introspection_texts(introspection_result)
    db = FAISS.from_documents(introspection_texts, cached_embedder)
    return db


async def aintrospection_db(endpoint) -> VectorStore:
    """
    This function is used to produce a vector store that
    will be used as context for the graphql query
    :param endpoint: The endpoint that will be used to produce the vector store
    :return: A vector store that will be used as context for the graphql query
    """
    introspection_result = await aintrospect(endpoint)
    introspection_texts = get_introspection_texts(introspection_result)
    db = await FAISS.afrom_documents(introspection_texts, cached_embedder)
    return db
