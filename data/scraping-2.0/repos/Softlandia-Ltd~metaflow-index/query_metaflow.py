"""Read the Metaflow-docs repository and create an index.

Must have the following env variables set:
    OPENAI_API_KEY
    GITHUB_TOKEN (if creating the index from scratch)
Alternatively can read them from a .env file, if present.
"""

import os
import logging

import plac
from llama_index import GPTFaissIndex, Document
from llama_index.readers import GithubRepositoryReader
from llama_index import LangchainEmbedding, LLMPredictor
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAIChat, OpenAI
import dotenv
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INDEX_STRUCT = "metaflow_index.json"
INDEX_VECTORS = "metaflow_vectors.dat"

# Read env variables from .env file, in case they are not set earlier
if not dotenv.load_dotenv():
    logger.warning("Could not load .env file")


def create_index(embed_model: LangchainEmbedding, chunk_size: int=256):
    """Create index from scratch.

    Args:
        embed_model: Embedding model to use for encoding documents.
        chunk_size: Length of individual encoded text segments, that will be
        used a context in queries. Larger values may contain more information
        but be harder to match to user requests.
    """
    logger.info("Creating index from scratch")
    reader = GithubRepositoryReader(
        "Netflix",
        "metaflow-docs",
        ignore_directories=["src", ".github", "static"],
        verbose=True,
    )
    documents = reader.load_data(branch="master")
    logging.info("Loaded %s documents", len(documents))

    # Create a Faiss instance
    embedding_len = len(embed_model._get_query_embedding("test"))
    faiss_index = faiss.IndexFlatL2(embedding_len)
    logger.debug("Embedding length: %s", embedding_len)
    index = GPTFaissIndex(
        documents,
        faiss_index=faiss_index,
        chunk_size_limit=chunk_size,
        embed_model=embed_model,
    )
    index.save_to_disk(INDEX_STRUCT, faiss_index_save_path=INDEX_VECTORS)


@plac.opt("n_sources", "Number of sources to use", type=int)
def main(n_sources: int = 2):
    """Create index and run queries."""
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    if not os.path.exists(INDEX_STRUCT):
        create_index(embed_model)

    # Use the ChatGPT model
    llm = LLMPredictor(OpenAIChat(model_name="gpt-3.5-turbo"))
    # Davinci is much more capable, but also much slower and more expensive
    # llm = LLMPredictor(OpenAI())
    index = GPTFaissIndex.load_from_disk(
        INDEX_STRUCT, faiss_index_save_path=INDEX_VECTORS, llm_predictor=llm,
        embed_model=embed_model
    )

    while True:
        # Take user input
        print("=== new query ===")
        query = input("Enter query: ")
        response = index.query(query, similarity_top_k=n_sources)
        print(response)
        print(response.source_nodes)


if __name__ == "__main__":
    plac.call(main)
