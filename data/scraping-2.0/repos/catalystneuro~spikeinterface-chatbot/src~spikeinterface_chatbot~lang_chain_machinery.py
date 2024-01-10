from pathlib import Path
from typing import Optional, List, Set, Tuple

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain.schema import Document


from .database_services import retrieve_qdrant_database, build_question_and_answer_retriever


def query_documentation(query: str, retriever_chain: RetrievalQA) -> dict[str, Set[str]]:
    """
    Queries the current RetrievalQA for an answer to the question.
    Formats the answer and returns it to be front-end.

    The procedure under the hood is the following. The question is first sent to the vector store to find the
    related chunks. Those chunks as combined with the question in an `in-context-learning` or
    `retrieval-augmented-generation` manner and sent to the OpenAI language model.

    The language model then generates an answer which is then provided as an output.

    Parameters
    ----------
    query : str
        A question described in natural language.

    Returns
    -------
    dict
        An answer to the query question.
    """

    chain_response = retriever_chain(query)
    answer_to_query = chain_response["result"]

    # Get link to sources
    source_documents = chain_response["source_documents"]
    context_documents = [document.page_content for document in source_documents]
    web_links = return_links_from_sources(source_documents)

    query_response = dict(answer_to_query=answer_to_query, web_links=web_links, context_documents=context_documents)
    return query_response


def return_links_from_sources(source_documents: List[Document]) -> Set[str]:
    """
    Converts local links in the source documents' metadata to web URLs.

    The function iterates through the metadata of each source document, extracts the local link, and
    converts it into a web URL using the `transform_local_link_to_web_url` function.

    Parameters
    ----------
    source_documents : List[Document]
        A list of Document objects containing the metadata with local links.

    Returns
    -------
    Set[str]
        A set of web URLs converted from the local links in the source documents' metadata.
    """

    source_metadata = (source.metadata for source in source_documents)
    local_links_to_documentation = (metadata["source"] for metadata in source_metadata)
    web_links_to_documentation = {transform_local_link_to_web_url(link) for link in local_links_to_documentation}

    return web_links_to_documentation


def transform_local_link_to_web_url(local_link: str) -> str:
    """
    Transforms a local link into a web URL.

    This function takes a local link, removes the local directory location, and prepends "https://" to create
    a web URL.

    Parameters
    ----------
    local_link : str
        A local link to be transformed into a web URL.

    Returns
    -------
    str
        A web URL created from the input local link.
    """
    local_directory_location = "rtdocs"
    local_link = local_link.split(local_directory_location)[1][1:]  # Remove leading slash
    web_link = f"https://{local_link}"

    return web_link
