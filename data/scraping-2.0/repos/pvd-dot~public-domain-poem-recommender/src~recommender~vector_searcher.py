"""Module for querying a vector search index containing poem embeddings."""
import collections
import openai
import numpy as np
from datasets import load_dataset
import dotenv
import os

EMBEDDING_DATA_SET = "pvd-dot/public-domain-poetry-with-embeddings"
MODEL = "text-embedding-ada-002"

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

Poem = collections.namedtuple(
    "Poem", ["id", "title", "author", "text", "views", "about", "birth_and_death_dates"]
)


class VectorSearch:
    """A class for performing vector search on a dataset of poem embeddings.

    This class utilizes a OpenAI's Ada text embedding model to convert text
    queries into embeddings and performs similarity searches in a dataset of
    poem embeddings. The poem embeddings encode the poem itself in addition
    to several metadata fields (author, bio, etc.), also using Ada.

    The class leverages the FAISS (Facebook AI Similarity Search) library for
    efficient similarity searching in high-dimensional spaces, making it
    suitablefor quick and relevant retrieval from a large collection of
    38k poems in the public domain."""

    def __init__(self):
        self.client = openai.OpenAI()
        self.data = load_dataset(EMBEDDING_DATA_SET, split="train")
        self.data.add_faiss_index(column="embedding")

    def convert_to_poem(self, id_):
        row = self.data[int(id_)]
        return Poem(
            id=row["id"],
            title=row["Title"],
            author=row["Author"],
            text=row["Poem Text"],
            views=row["Views"],
            about=row["About"],
            birth_and_death_dates=row["Birth and Death Dates"],
        )

    def search(self, query_text, limit=1):
        query_embedding = np.array(
            self.client.embeddings.create(input=[query_text], model=MODEL)
            .data[0]
            .embedding
        )
        _, results = self.data.search("embedding", query_embedding, k=limit)
        return [self.convert_to_poem(id) for id in results]
