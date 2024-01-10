import sys
from langchain_community.embeddings.openai import OpenAIEmbeddings
import json
import numpy as np
from numpy.linalg import norm
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


"""
required environment variables:
OPENAI_API_KEY

the scripts expects to be given filepaths to documents as args

document structure(should be a json file):
{
    "content": "...",
    "key": "..." // a memory key from memory schema
}
"""


def main():
    vectors = []
    for arg in sys.argv[1:]:
        print("opening ", arg)
        with open(arg) as json_data:
            document = json.load(json_data)
            vectors.append(OpenAIEmbeddings().embed_query(document["content"]))

    print(
        "Cosine similarity for these documents: ",
        cosine_similarity(vectors[0], vectors[1]),
    )


if __name__ == "__main__":
    main()
