from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
import os


root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_CACHE_PATH = os.path.join(root_dir, "_cache", "_embedding_cache")


def getEmbeddingsProvider(provider):

    embedding_cache_store = LocalFileStore(
        os.path.join(EMBEDDING_CACHE_PATH, provider))

    if provider == 'openai':
        openApiKey = os.environ.get('OPENAI_API_KEY')
        underlying_embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)
    else:
        raise NotImplementedError(
            f'Provider {provider} not implemented')

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, embedding_cache_store, namespace=underlying_embeddings.model
    )

    return cached_embedder
