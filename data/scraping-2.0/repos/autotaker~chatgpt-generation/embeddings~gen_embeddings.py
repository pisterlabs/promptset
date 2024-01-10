import os
import sys
from openai.embeddings_utils import get_embedding
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def gen_embedding(file: str):
    with open(file) as f:
        content = f.read()
    embedding = get_embedding(content, "text-embedding-ada-002")
    with open(file + ".encoded", "w") as f:
        f.write(repr(embedding))


if __name__ == "__main__":
    for file in sys.argv[1:]:
        print("get_embedding:", file)
        gen_embedding(file)
