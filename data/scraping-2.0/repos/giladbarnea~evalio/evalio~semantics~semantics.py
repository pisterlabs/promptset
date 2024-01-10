from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import openai
from openai.embeddings_utils import (cosine_similarity,
                                     distances_from_embeddings)
from openai.embeddings_utils import get_embedding as openai_get_embedding
from openai.embeddings_utils import indices_of_nearest_neighbors_from_distances

if TYPE_CHECKING:
    import numpy
    import pandas

DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str | "numpy.ndarray[float]") -> "numpy.ndarray[float]":
    import numpy as np

    if isinstance(text, np.ndarray):
        return text
    # We need to decode it ourselves, see https://github.com/EliahKagan/embed-encode
    base64_encoded = openai_get_embedding(text, engine=DEFAULT_EMBEDDING_MODEL, encoding_format="base64")
    buffer = base64.b64decode(base64_encoded)
    return np.frombuffer(buffer, dtype=np.float32)


def get_semantic_distance(
    string_or_embedding_1: "numpy.ndarray[float]" | str, string_or_embedding_2: "numpy.ndarray[float]" | str
) -> int:
    string_1_embedding = get_embedding(string_or_embedding_1)
    string_2_embedding = get_embedding(string_or_embedding_2)
    similarity = cosine_similarity(string_1_embedding, string_2_embedding)
    distance = 1 - similarity
    return distance


def get_nearest_neighbors(
    strings: list[str | "numpy.ndarray[float]"], index_of_query_string: int, k_nearest_neighbors: int = 5
):
    embeddings = [get_embedding(string) for string in strings]
    query_embedding = embeddings[index_of_query_string]
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    query_string = strings[index_of_query_string]
    print(f"Source string: {query_string}")
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        print(
            f"""
            --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
            String: {strings[i]}
            Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors


def embed_column(column: "pandas.Series") -> "pandas.Series":
    return column.apply(lambda x: get_embedding(x))


def get_similarity_to_column(
    column: "pandas.Series", string_or_embedding: "numpy.ndarray[float]" | str
) -> "pandas.Series":
    if isinstance(string_or_embedding, str):
        embedding = get_embedding(string_or_embedding)
    return column.apply(lambda x: cosine_similarity(x, embedding))


def create_clusters_from_column(column: "pandas.Series", *, n_clusters: int, matrix: np.ndarray = None):
    from sklearn.cluster import KMeans

    if matrix is None:
        matrix = np.vstack(column.values)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=42)
    kmeans.fit(matrix)
    return kmeans.labels_


def create_clusters_from_embeddings(
    df: "pandas.DataFrame", *, embeddings_column, target_cluster_column, n_clusters: int
) -> np.ndarray:
    matrix = np.vstack(df[embeddings_column].values)
    labels = create_clusters_from_column(df[embeddings_column], n_clusters=n_clusters, matrix=matrix)
    df[target_cluster_column] = labels
    return matrix


def visualize_embeddings(df: "pandas.DataFrame", *, embeddings_column, score_column, title=None):
    from ast import literal_eval

    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    matrix = np.array(df[embeddings_column].apply(literal_eval).to_list())
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]
    color_indices = df[score_column].values - 1
    colormap = matplotlib.colors.ListedColormap(colors)
    plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    if title:
        plt.title(title)
    plt.show()


def visualize_embedding_clusters(
    df: "pandas.DataFrame", *, embeddings_column, target_cluster_column, score_column, n_clusters: int, title=None
):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    matrix = create_clusters_from_embeddings(
        df, embeddings_column=embeddings_column, target_cluster_column=target_cluster_column, n_clusters=n_clusters
    )

    print(df.groupby(target_cluster_column)[score_column].mean().sort_values())

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    colors = ["blue", "red", "green", "purple", "orange", "cyan", "yellow", "magenta"]

    for category, color in enumerate(colors[:n_clusters]):
        xs = np.array(x)[df[target_cluster_column] == category]
        ys = np.array(y)[df[target_cluster_column] == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    if title:
        plt.title(title)
    plt.show()


CLASSIFY_REVIEWS_PROMPT_TEMPLATE = '''What do the following customer reviews have in common?

Customer reviews:
"""
{reviews}
"""

Theme:
'''


def classify_clusters_with_llm(
    df: "pandas.DataFrame",
    *,
    cluster_column,
    n_clusters: int,
    completion_model="text-davinci-003",
    prompt_template=CLASSIFY_REVIEWS_PROMPT_TEMPLATE,
):
    """Not generic. Only works for the Amazon reviews dataset."""
    review_samples_per_cluster = 5

    for i in range(n_clusters):
        reviews = "\n".join(
            df[df[cluster_column] == i]
            .combined.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .sample(review_samples_per_cluster, random_state=42)
            .values
        )
        response = openai.Completion.create(
            engine=completion_model,
            prompt=prompt_template.format(reviews=reviews),
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(f"\x1b[97mCluster {i} Theme:\x1b[0m " + response["choices"][0]["text"].replace("\n", ""), end="\n\n")

        sample_cluster_rows = df[df[cluster_column] == i].sample(review_samples_per_cluster, random_state=42)
        longest_summary_len = sample_cluster_rows.Summary.str.len().max()
        for j in range(review_samples_per_cluster):
            print(sample_cluster_rows.Score.values[j], end=", ")
            print(sample_cluster_rows.Summary.values[j], end="  ")
            print(" " * (longest_summary_len - len(sample_cluster_rows.Summary.values[j])), end=" ")
            print(sample_cluster_rows.Text.str[:70].values[j])

        print("─" * 100)


def search_reviews(
    df: "pandas.DataFrame", product_description: str, *, embeddings_column, top_n=3, pprint=True
) -> "pandas.Series":
    """Not generic. Only works for the Amazon reviews dataset."""
    df["similarity"] = get_similarity_to_column(df[embeddings_column], product_description)

    results = (
        df.sort_values("similarity", ascending=False)
        .head(top_n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200], end="\n\n")
    return results
