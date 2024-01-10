from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings
)

from langchain.schema.embeddings import Embeddings

# Should use $ENV:EMBEDDING_MODEL to choose an embedding model
# dimensions are used by vector index and is model specific
def load_embeddings(model: str, config: dict) -> tuple[Embeddings, int]:
    if model == 'llama2':
        embeddings = OllamaEmbeddings(
            base_url=config["base_url"],
            model=model
        )
        dimensions = 4096
    else: # sentence_transformer
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="/embedding_model"
        )
        dimensions = 384

    return embeddings, dimensions