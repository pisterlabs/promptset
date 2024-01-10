from loguru import logger
from langchain.chat_models import ChatOpenAI

from ai_driver.langsmith_config import get_client
from ai_driver.vector_storage.faiss_manager import embed_FAISS_from_documents
from ai_driver.vector_storage.local_loader import get_default_local_download
from ai_driver.retrieval.qa import qa_pipeline
from ai_driver.instruct import InstructConfig


def local_download_pipeline(dir_path: str, embed_model: str):
    """Example local store pipeline"""

    logger.info("Local download pipeline: FAISS")
    texts = get_default_local_download(dir_path=dir_path)
    config: InstructConfig = InstructConfig(embed_model=embed_model)

    embedding_model_name = config.embed_model
    embedding_model_kwargs = {"device": "cuda"}

    vector_store = embed_FAISS_from_documents(
        texts, embedding_model_name, embedding_model_kwargs
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    model: ChatOpenAI = ChatOpenAI(
        client=get_client(),
        temperature=0.0,
        model="gpt-3.5-turbo-0613",
    )
    response = qa_pipeline("How do saving throws work?", retriever, model)
    return response
