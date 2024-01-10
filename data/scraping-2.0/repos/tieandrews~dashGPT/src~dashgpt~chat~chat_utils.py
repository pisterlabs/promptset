# Author: Ty ANdrews
# Date: 2023-09021
import os

import openai
import platform
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import re

from dashgpt.logs import get_logger
from dashgpt.data.langchain_utils import count_tokens

# for eployment on azure, Chroma SQlite version is out oof date, over write
# inspired from: https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
if platform.system() == "Linux":
    # these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import Chroma

logger = get_logger(__name__)

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


def connect_to_vectorstore():
    """
    Connect to the VectorStore and return a VectorStore object.

    Returns
    -------
    VectorStore object
        The VectorStore object connected to the VectorStore.
    """
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    chroma_db = Chroma(
        persist_directory = "data/processed/reddit_jokes_chroma_db",
        embedding_function = embedding_function,
        collection_name = "reddit_jokes_2000"
    )

    return chroma_db


def stream_send_messages(prompt):
    """
    Send a prompt to the OpenAI API and return the response.

    Parameters
    ----------
    prompt : str
        The prompt to send to the OpenAI API.

    Returns
    -------
    OpenAI ChatCompletion object
        The response from the OpenAI API.
    """

    # calculate the number of tokens by combining the prompt and the user prompt
    total_tokens = 0
    for message in prompt:
        total_tokens += count_tokens(message["content"])

    if total_tokens > 2048:
        logger.warning(
            f"Total tokens {total_tokens} exceeds maximum of 2048. Using turbo 16k context model."
        )
        # uncomment this to auto flip to 16k model
        # model = "gpt-3.5-turbo-16k"
        # instead only use the last 512 tokens of user prompt to limit abuse
        prompt[-1]["content"] = prompt[-1]["content"][-512:]
    else:
        model = "gpt-3.5-turbo"

    """OpenAI API call. You may choose parameters here but `stream=True` is required."""
    return openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        stream=True,
        max_tokens=1024,
        temperature=0.5,
    )

def get_relevant_documents(
    user_prompt,
    vector_store,
    k=3,
    method="similarity",
):
    """
    Get the most relevant documents from the VectorStore for a given user prompt.

    Parameters
    ----------
    user_prompt : str
        The user prompt to search for in the VectorStore.
    vector_store : Zilliz object
        The object connected to the VectorStore.
    method: str, optional
        The method to use for searching the VectorStore, options are mmr, similarity. Default is "similarity".

    Returns
    -------
    list of Document objects
        The list of relevant documents from the VectorStore.
    """

    if method == "mmr":
        relevant_documents = vector_store.max_marginal_relevance_search(
            query=user_prompt,
            k=k,
            fetch_k=10,
        )

        return relevant_documents
    elif method == "similarity":
        relevant_documents = vector_store.similarity_search_with_score(
            query=user_prompt,
            k=k,
        )
        # take the relavant documents which is a list of tuples of Document, score and convert to a list of Document
        # with a new field in metadata of each document called score
        for doc, score in relevant_documents:
            doc.metadata["score"] = score

        # only keep the documents from the relevant documents tuples, not score
        relevant_documents_with_score = [doc for doc, score in relevant_documents]

        # return selected_relevant_documents
        return relevant_documents_with_score
    else:
        raise ValueError("method must be mmr or similarity")


def convert_documents_to_chat_context(relevant_documents):
    """
    Convert a list of relevant documents to a chat context string.

    Parameters
    ----------
    relevant_documents : list of Document objects
        The list of relevant documents to convert.

    Returns
    -------
    str
        The chat context string created from the relevant documents.
    """
    # combine the page content from the relevant documents into a single string
    context_str = ""
    for i, doc in enumerate(relevant_documents):
        context_str += f"{doc.page_content}\n"

    return context_str


def convert_chat_history_to_string(
    chat_history: dict, include_num_messages: int = 1, questions_only = False
) -> str:
    """
    Convert a chat history dictionary to a string.

    Parameters
    ----------
    chat_history : dict
        A dictionary containing the chat history.

    Returns
    -------
    str
        A string representation of the chat history.

    Notes
    -----
    The chat history dictionary should have the following format:
    {
        "chat_history": [
            {
                "role": "user" or "assistant",
                "content": "message content"
            },
            ...
        ]
    }
    The returned string will have the following format:
    "user: message content\nassistant: message content\n..."

    """
    if questions_only is False:
        start_index = -(2 * include_num_messages) - 1
        chat_history_str = ""
        for line in chat_history["chat_history"][start_index:-1]:
            chat_history_str += f"{line['role']}: {line['content'].strip()}\n"

        logger.debug(f"Chat history: {chat_history_str}")
    elif questions_only is True:
        start_index = -(2 * include_num_messages) - 1
        chat_history_str = ""
        for line in chat_history["chat_history"][start_index:-1]:
            if line['role'] == 'user':
                chat_history_str += f"{line['role']}: {line['content'].strip()}\n"

        logger.debug(f"Chat history: {chat_history_str}")

    return chat_history_str

