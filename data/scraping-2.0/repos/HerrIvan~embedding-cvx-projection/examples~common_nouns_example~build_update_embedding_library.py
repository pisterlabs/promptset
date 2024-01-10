"""
In this example we will get the embeddings of some common Nouns and will try to play the game of
explaining one in terms of others. For instance, King = Monarch + Man.

We will use the openai API to get the embeddings of the words.
"""
import os

from dotenv import load_dotenv


from embedding_library import EmbeddingLibrary
from embedding_library.embedders import OpenAIEmbedder
from embedding_library.common_nouns import common_nouns

load_dotenv()

EMBEDDING_LIBRARY_PATH = "embedding_library/common_nouns.pkl"


def main():
    # get the embeddings of the common nouns

    # print local directory
    print(os.getcwd())

    if os.path.exists(EMBEDDING_LIBRARY_PATH):
        embs = EmbeddingLibrary.load(EMBEDDING_LIBRARY_PATH)

        print("Updating embedding library...")
        # update the library with new embeddings
        embs.add_embeddings(list(set(common_nouns)))

    else:
        print("Creating embedding library...")

        embs = EmbeddingLibrary(
            embedder=OpenAIEmbedder(model_name="text-embedding-ada-002"),
            initial_inputs=common_nouns
        )

    embs.save(EMBEDDING_LIBRARY_PATH)


if __name__ == "__main__":
    main()
