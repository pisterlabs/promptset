"""Read the bytewax repository and create an index.

Must have the following env variables set:
    OPENAI_API_KEY
    GITHUB_TOKEN (if creating the index from scratch)
"""

import os
import logging

import plac
from llama_index import GPTSimpleVectorIndex
from llama_index.readers import GithubRepositoryReader
from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INDEX_FILE = "bytewax_index.json"


@plac.opt("n_sources", "Number of sources to use", type=int)
def main(n_sources: int = 2):
    """Create index and run queries."""
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # TODO: try FAISS index
    if os.path.exists(INDEX_FILE):
        logger.info("Loading index from file")
        index = GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)
    else:
        logger.info("Creating index from scratch")
        reader = GithubRepositoryReader(
            "bytewax",
            "bytewax",
            ignore_directories=[".github", "migrations", "src"],
            verbose=False,
        )
        documents = reader.load_data(branch="main")
        print(len(documents))
        print(documents[0])
        logging.info("Documents loaded. Creating index")
        index = GPTSimpleVectorIndex(
            documents, chunk_size_limit=512, embed_model=embed_model
        )
        index.save_to_disk(INDEX_FILE)

    while True:
        query = input("Enter query: ")
        results = index.query(
            query, similarity_top_k=n_sources, embed_model=embed_model
        )
        print(results)


if __name__ == "__main__":
    plac.call(main)
