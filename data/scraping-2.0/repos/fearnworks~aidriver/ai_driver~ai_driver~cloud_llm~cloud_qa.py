import os
from loguru import logger
from ai_driver.vector_storage.pinecone_manager import (
    get_default_pinecone_session,
    PineconeConfig,
)
from langchain.chat_models import ChatOpenAI
from ai_driver.langsmith_config import get_client
from ai_driver.retrieval.qa import qa_pipeline


def pinecone_qa_pipeline(query: str):
    """Example Pinecone Pipeline"""
    logger.info("Pinecone Pipeline")
    config = PineconeConfig(
        openai_key=os.getenv("OPENAI_API_KEY"),
        pinecone_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_API_ENV"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
    )
    logger.info(config)
    vector_store = get_default_pinecone_session(config).docsearch
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    model: ChatOpenAI = ChatOpenAI(
        client=get_client(),
        temperature=0.0,
        model="gpt-3.5-turbo-0613",
    )
    response = qa_pipeline(query, retriever, model)
    return response
