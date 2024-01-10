from langchain.chat_models import ChatOpenAI

from app.chat.models import ChatArgs
from app.chat.vector_stores.pinecone import build_retriever


def build_llm(chat_args: ChatArgs, model_name: str) -> ChatOpenAI:
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.
    """
    return ChatOpenAI(streaming=chat_args.streaming, model=model_name)
