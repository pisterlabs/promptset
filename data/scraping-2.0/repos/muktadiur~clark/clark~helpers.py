import os
from typing import List

from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain
)

from clark.base import BaseConversation
from clark.document import process_documents
from clark.gpt4all import GPT4AllConversation
from clark.hf import HFConversation
from clark.openai import OpenAIConversation


def get_converstation() -> BaseConversation:
    if os.getenv("CONVERSATION_ENGINE") == "gpt4all":
        return GPT4AllConversation()

    if os.getenv("CONVERSATION_ENGINE") == "hf":
        return HFConversation()

    return OpenAIConversation()


def create_vectors() -> None:
    texts: List[str] = process_documents()
    get_converstation().create_store(texts=texts)


def get_chain() -> BaseConversationalRetrievalChain:
    try:
        return get_converstation().get_chain()
    except FileNotFoundError:
        create_vectors()
        return get_converstation().get_chain()
