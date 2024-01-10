import pinecone
from pinecone import UnauthorizedException

from ..config.config import get_config
from ..utils import logger
from .chromadb import ChromaDBVectorStore, build_chroma_client
from .embedding.base import BaseEmbedding, EmbeddingsType
from .embedding.gradient import GradientEmbedding
from .lancedb import LanceDBVectorStore
from .milvus import MilvusVectorStore
from .pinecone import PineconeVectorStore
from .base import VectorStoreType
from .embedding import OpenAiEmbedding, CohereEmbedding, HuggingfaceBgeEmbedding, PalmEmbedding
from .qdrant import QdrantVectorStore, create_qdrant_client
from .redis import RedisVectorStore
from .weaviate import create_weaviate_client, WeaviateVectorStore


class VectorFactory:

    @classmethod
    def get_vector_storage(cls, type: VectorStoreType | str, embedding_model: BaseEmbedding, index_name: str, api_key: str | None = None):

        if isinstance(type, str):
            type = VectorStoreType.get_type(type.lower())

        if type == VectorStoreType.PINECONE:
            try:
                api_key = get_config("PINECONE_API_KEY")
                env = get_config("PINECONE_ENVIRONMENT")
                if api_key is None or env is None:
                    raise ValueError("Pinecone API key not found")
                pinecone.init(api_key=api_key, environment=env)

                if index_name not in pinecone.list_indexes():
                    sample_embedding = embedding_model.get_embedding("sample")
                    if "error" in sample_embedding:
                        logger.error(
                            f"Error in embedding model {sample_embedding}")

                    # if does not exist, create index
                    pinecone.create_index(
                        index_name,
                        dimension=len(sample_embedding),
                        metric='dotproduct'
                    )
                index = pinecone.Index(index_name)
                return PineconeVectorStore(index=index, namespace=env, embedding_model=embedding_model)
            except UnauthorizedException:
                raise ValueError("Pinecone API key not found")

        if type == VectorStoreType.WEAVIATE:
            url = get_config("WEAVIATE_URL")
            api_key = get_config("WEAVIATE_API_KEY")
            if api_key is None or url is None:
                raise ValueError(
                    f"API_KEY for {type} not Provided in both request and env")

            client = create_weaviate_client(
                url=url,
                api_key=api_key
            )
            return WeaviateVectorStore(client=client, index_name=index_name, embedding_model=embedding_model)

        if type == VectorStoreType.QDRANT:
            client = create_qdrant_client(api_key=api_key)
            return QdrantVectorStore(client=client, collection_name=index_name, embedding_model=embedding_model)

        if type == VectorStoreType.REDIS:
            redis = RedisVectorStore(index_name, embedding_model)
            return redis

        if type == VectorStoreType.LANCEDB:
            db_path = get_config("LANCE_DB_PATH") or "./lance"
            lance = LanceDBVectorStore(
                db_path=db_path, api_key=api_key, embeddings=embedding_model, index_name=index_name
            )
            return lance

        if type == VectorStoreType.CHROMADB:
            chroma_host_name = get_config("CHROMA_HOST_NAME")
            chroma_port = get_config("CHROMA_PORT")

            use_persistance = False
            if chroma_port is None or chroma_host_name is None:
                use_persistance = True

            client = build_chroma_client(
                host=chroma_host_name, port=chroma_port, persistance=use_persistance)
            chroma = ChromaDBVectorStore(
                client_options=client, collection_name=index_name, embedding_model=embedding_model)
            return chroma

        if type == VectorStoreType.MILVUS:
            milvus_host_name = get_config("MILVUS_HOST") or "localhost"
            milvus_port = get_config("MILVUS_PORT") or "19530"

            chroma = MilvusVectorStore(
                host=milvus_host_name,
                port=milvus_port,
                collection_name=index_name,
                embedding_model=embedding_model,
            )
            return chroma

        raise ValueError(f"Vector store {type} not supported")

    @classmethod
    def get_embeddings(cls, type: EmbeddingsType | str, api_key: str | None = None, model: str | None = None):
        embedding_model: BaseEmbedding

        if isinstance(type, str):
            type = EmbeddingsType.get_type(type.lower())

        if type == EmbeddingsType.COHERE:
            api_key = api_key or get_config("COHERE_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"API_KEY for {type} not Provided in both request and env")
            embedding_model = CohereEmbedding(
                api_key=api_key, model=model or "command-large")

        elif type == EmbeddingsType.OPENAI:
            api_key = api_key or get_config("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"API_KEY for {type} not Provided in both request and env")
            embedding_model = OpenAiEmbedding(
                api_key=api_key, model=model or "text-embedding-ada-002")

        elif type == EmbeddingsType.PALM:
            api_key = api_key or get_config("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"API_KEY for {type} not Provided in both request and env")
            embedding_model = PalmEmbedding(
                api_key=api_key, model=model or "models/text-bison-001")

        elif type == EmbeddingsType.GRADIENT:
            gradient_access_token = get_config("GRADIENT_ACCESS_TOKEN")
            gradient_workspace_id = get_config("GRADIENT_WORKSPACE_ID")

            if gradient_workspace_id is None or gradient_access_token is None:
                raise ValueError(
                    f"Gradient Access Token Not Provided : {gradient_access_token}")

            embedding_model = GradientEmbedding(
                gradient_access_token=gradient_access_token,
                gradient_workspace_id=gradient_workspace_id,
                model=model or "bge-medium",
            )

        elif type == EmbeddingsType.HUGGINGFACE:
            embedding_model = HuggingfaceBgeEmbedding(
                model=model or "BAAI/bge-small-en")

        else:
            raise ValueError(f"Vector store {type} not supported")

        # sample_embedding = embedding_model.get_embedding("sample")
        # if "error" in sample_embedding:
        #     logger.error(f"Error in embedding model {sample_embedding}")
        return embedding_model
