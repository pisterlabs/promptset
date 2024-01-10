from langchain.document_loaders import GCSDirectoryLoader, PDFPlumberLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.vertexai import VertexAI

from aitestdrive.common.config import config
from aitestdrive.common.logging import log_to_console
from aitestdrive.persistence.qdrant import QdrantService


def create_llm():
    return VertexAI(
        temperature=0.1,
        max_output_tokens=256,
        top_p=0.8,
        top_k=40
    ).with_config(log_to_console())


def create_embeddings():
    return VertexAIEmbeddings()


def create_gcs_directory_loader():
    def loader_func(file_path: str):
        return PDFPlumberLoader(file_path)

    return GCSDirectoryLoader(project_name=config.google_cloud_project,
                              bucket=config.document_bucket,
                              loader_func=loader_func)


def create_qdrant_service():
    return QdrantService(create_embeddings())
