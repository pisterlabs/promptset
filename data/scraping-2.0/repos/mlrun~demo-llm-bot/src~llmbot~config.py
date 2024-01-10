import logging

from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from pydantic import BaseSettings, PyObject


class AppConfig(BaseSettings):
    # Embeddings
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_chunk_size: int = 500
    embeddings_chunk_overlap: int = 50
    embeddings_class: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"
    embeddings_encode_kwargs: dict = {"batch_size": 32}

    # Vector store
    vector_store_class: PyObject = "langchain.vectorstores.Milvus"
    vector_store_connection_args: dict = {
        "host": "my-release-milvus.default.svc.cluster.local",
        "port": "19530",
    }

    # LLM
    llm_class: PyObject = "langchain.chat_models.openai.ChatOpenAI"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0

    # Nuclio functions store their code in a specific directory
    repo_dir: str = "/opt/nuclio"

    # Helpers
    def get_embedding_function(self) -> Embeddings:
        return self.embeddings_class(
            model_name=self.embeddings_model,
            encode_kwargs=self.embeddings_encode_kwargs,
        )

    def get_vector_store(self) -> VectorStore:
        return self.vector_store_class(
            embedding_function=self.get_embedding_function(),
            connection_args=self.vector_store_connection_args,
        )

    def get_llm(self) -> BaseChatModel:
        return self.llm_class(model=self.llm_model, temperature=self.llm_temperature)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("llmbot")
