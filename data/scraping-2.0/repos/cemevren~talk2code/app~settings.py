from pydantic_settings import BaseSettings
from langchain.text_splitter import Language


class Settings(BaseSettings):
    openai_api_key: str = ""
    tokenizers_parallelism: bool = False
    llm_model: str = "gpt-4-1106-preview"
    qdrant_url: str = ""
    qdrant_collection_name: str = "langchain"
    qdrant_collection_dim: int = 512
    vector_search_type: str = "mmr"  # "similarity"
    vector_search_k: int = 4
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_model_device: str = "mps"  # mps, cpu, cuda https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    embedding_model_size: int = 1024
    loader_exclude_paths: list[str] = ["examples/**"]
    loader_language: Language = Language.PYTHON
    loader_suffixes: list[str] = [".py"]


settings = Settings()
