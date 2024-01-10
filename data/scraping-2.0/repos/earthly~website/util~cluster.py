import os
import openai
from sklearn.cluster import KMeans
from typing import List, Tuple
import pickle
import numpy as np

CACHE_FILE = "./util/cluster_embeddings_cache.pkl"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

cache = load_cache()

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)


# 1. Load the Markdown Files
def load_markdown_files(folder_path: str) -> Tuple[List[str], List[str]]:
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # all_files = all_files[:100]
    markdown_texts = []

    for file in all_files:
        print(file)
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            markdown_texts.append(f.read())

    return all_files, markdown_texts

def get_embedding(text: str, filename : str, model="text-embedding-ada-002") -> List[float]:
    if filename in cache:
        return cache[filename]

    print(f"get embedding:{filename}")
    while True:
        try:
            # text = text.replace("\n", " ")
            embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
            cache[filename] = embedding
            return embedding
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                print(f"{filename}: dropping lines")
                lines = text.split("\n")
                if len(lines) > 10:
                    text = "\n".join(lines[:-10])  # Drop last 10 lines
                else:
                    print("Less than 10 left")
                    # If the document has fewer than 10 lines, return an empty embedding
                    return [0.0] * 1536
            else:
                raise e
                
def generate_embeddings(markdown_texts: List[str],  filenames: List[str]) -> List[List[float]]:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    embeddings = [get_embedding(text, filename) for text, filename in zip(markdown_texts, filenames)]
    return embeddings


def handle_nan_in_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    cleaned_embeddings = []
    for embedding in embeddings:
        if np.isnan(np.sum(embedding)):
            cleaned_embedding = [0 if np.isnan(val) else val for val in embedding]
            cleaned_embeddings.append(cleaned_embedding)
        else:
            cleaned_embeddings.append(embedding)
    return cleaned_embeddings

# 3. Clustering
def cluster_embeddings(embeddings: List[List[float]], n_clusters: int) -> List[int]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_

# 4. Display the Clusters
def display_clusters(clusters: List[List[str]]) -> None:
    for i, cluster in enumerate(clusters):
        if len(cluster) < 3:
            continue
        print(f"Cluster {i + 1}:")
        for file_path in cluster:
            print(f"  - {file_path}")

def main() -> None:
    try:
        folder_path = "./blog/_posts"
        n_clusters = 50  # Adjust as necessary

        all_files, markdown_texts = load_markdown_files(folder_path)
        embeddings = generate_embeddings(markdown_texts, all_files)
        embeddings = handle_nan_in_embeddings(embeddings)
        labels = cluster_embeddings(embeddings, n_clusters)

        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(all_files[idx])
        
        display_clusters(clusters)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save the updated cache
        save_cache(cache)


if __name__ == "__main__":
    main()
