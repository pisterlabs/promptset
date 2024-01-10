from pathlib import Path as P
from typing import List, Optional

import yaml
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


def load_model_from_yaml(base_model_cls, yaml_path):
    with open(yaml_path) as f:
        raw_obj = yaml.safe_load(f)
    return base_model_cls.parse_obj(raw_obj)


class LoaderConfig(BaseModel):

    path: str
    loader_type: str
    glob_pattern: str = Field(default="**/*")

    def get_index_name(self):
        return f"{P(self.path).name}-{self.loader_type}"


class EmbeddingConfig(BaseModel):

    embedding_model_name: str
    timeout: int

    @staticmethod
    def get_default():
        return EmbeddingConfig(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2", timeout=180
        )

    def load_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)


class PreprocessingConfig(BaseModel):

    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def get_default():
        return PreprocessingConfig(chunk_size=512, chunk_overlap=128)


class PersistenceConfig(BaseModel):
    index_type: str
    persist_directory: str
    collection_name: Optional[str]
    distance_func: str
