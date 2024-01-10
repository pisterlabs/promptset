"""
Search your data in Azure Cognitive Search with a question, and get the relevant 
documents.
"""
import openai, os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, CognitiveSearchConnection


@tool
def get_context(
    question: str,
    azure_open_ai_connection: AzureOpenAIConnection,
    azure_search_connection: CognitiveSearchConnection,
    index_name: str,
    embedding_deployment: str,
) -> list[str]:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    openai.api_type = azure_open_ai_connection.api_type
    openai.api_base = azure_open_ai_connection.api_base
    openai.api_version = azure_open_ai_connection.api_version
    openai.api_key = azure_open_ai_connection.api_key

    query_vector = Vector(
        value=openai.Embedding.create(engine=embedding_deployment, input=question)[
            "data"
        ][0]["embedding"],
        fields="embedding",
    )

    search_client = SearchClient(
        endpoint=azure_search_connection.api_base,
        index_name=index_name,
        credential=AzureKeyCredential(azure_search_connection.api_key),
    )

    docs = search_client.search(search_text="", vectors=[query_vector], top=2)
    context = [doc["content"] for doc in docs]

    return context
