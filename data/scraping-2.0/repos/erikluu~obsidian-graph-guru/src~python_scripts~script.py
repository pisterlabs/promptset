import sys
import os
import json
import openai
import numpy as np
from sklearn.decomposition import PCA
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# tenacity helps with rate limits
# checkout here: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(10))
def createEmbedding(block):
    # returns array of size 1536
    return openai.Embedding.create(input = [block], model="text-embedding-ada-002")['data'][0]['embedding']

def preprocessText(text):
    pass

def getAllEmbeddings(path):
    # recursively walk through dir to get all markdown files
    # returns a dictionary where {filename1: content1, filename2: content2, ...}
    embeddings = {}
    # only getting a subset of the directories and files because it's a lot
    for item in os.listdir(path):
        content = os.path.join(path, item)
        if os.path.isdir(content):
            embeddings.update(getAllEmbeddings(content))
        elif content.endswith(".md"):
            fp = open(content)
            embeddings[item] = createEmbedding(fp.read())
            fp.close()
    return embeddings

def reduce_dimensions(vec_embeddings):
    pca = PCA(n_components=2)
    pca.fit(vec_embeddings)
    pca_encodings = pca.transform(vec_embeddings)
    return pca_encodings

def main():
    vault_path = sys.argv[1]
    openai.api_key = sys.argv[2]

    embeddings = getAllEmbeddings(vault_path)

    with open(f'{vault_path}/.obsidian/plugins/obsidian-graph-guru/results/embeddings.json', 'w') as fp:
        json.dump(embeddings, fp)

    with open(f'{vault_path}/.obsidian/plugins/obsidian-graph-guru/results/embeddings.json','r') as infile:
        data = json.load(infile)

    embedding_labels = list(data.keys())
    # embedding_size = len(data[embedding_labels[0]])
    embeddings = np.array([data[label] for label in embedding_labels])
    latent_space = reduce_dimensions(embeddings)
    coordinates = dict(zip(embedding_labels, latent_space.tolist()))
    with open(f'{vault_path}/.obsidian/plugins/obsidian-graph-guru/results/coordinates.json', 'w') as fp:
        json.dump(coordinates, fp)

    # with open(f'{vault_path}/.obsidian/plugins/obsidian-leaflet-plugin/data.json', 'w') as fp:
    #     json.dump(coordinates, fp)

    print("Done!")

if __name__ == "__main__":
    main()