# Importing the necessary libraries
from box import Box
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.vectorstores.faiss import FAISS

from language_models.language_models import LanguageModel
from utils.helpers import get_config

# Get the configuration
cfg: Box = get_config('conversation_chain.yaml')


def create_conversation_chain(vectorstore: FAISS, language_model: LanguageModel) -> BaseConversationalRetrievalChain:
    """
    Create a conversation chain which consists of initialized memory, retriever
    defined by the vectorstore, and a language model.

    Parameters
    ----------
    vectorstore: FAISS
        The vectorstore instance used to create the retriever.
    language_model: LanguageModel
        The language model instance used to set up the conversation chain.

    Returns
    -------
    conversation_chain: BaseConversationalRetrievalChain
        The conversation chain built using the given vectorstore and language model.
    """
    memory: ConversationBufferWindowMemory = initialize_memory()
    retriever: VectorStoreRetriever = create_retriever(vectorstore)
    conversation_chain: BaseConversationalRetrievalChain = build_conversation_chain(language_model, retriever, memory)
    return conversation_chain


def initialize_memory() -> ConversationBufferWindowMemory:
    """
    Initialize the ConversationBufferWindowMemory with certain parameters.

    Returns
    -------
    memory: ConversationBufferWindowMemory
        Initialized memory for the conversation chain.
    """
    return ConversationBufferWindowMemory(k=cfg.memory.k, memory_key='chat_history', return_messages=True)


def create_retriever(vectorstore: FAISS) -> VectorStoreRetriever:
    """
    Create a retriever with certain search parameters.

    Parameters
    ----------
    vectorstore: FAISS
        The vectorstore instance used to create the retriever.

    Returns
    -------
    retriever: VectorStoreRetriever
        A retriever created using the given vectorstore.
    """
    return vectorstore.as_retriever(search_kwargs={"k": cfg.retriever.k})


def build_conversation_chain(
        language_model: LanguageModel, retriever: VectorStoreRetriever, memory: ConversationBufferWindowMemory
) -> BaseConversationalRetrievalChain:
    """
    Build the ConversationalRetrievalChain using a given language model, retriever,
    and memory.

    Parameters
    ----------
    language_model: LanguageModel
        The language model instance used to set up the conversation chain.
    retriever: VectorStoreRetriever
        The retriever used in the conversation chain.
    memory: ConversationBufferWindowMemory
        The memory used in the conversation chain.

    Returns
    -------
    conversation_chain: BaseConversationalRetrievalChain
        The conversation chain built using the given language model, retriever and memory.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=retriever,
        memory=memory,
        chain_type=cfg.conversation_chain.chain_type,
        verbose=cfg.conversation_chain.verbose,
    )
