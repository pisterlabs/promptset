"""
innovation_sweet_spots.utils.cluster_analysis_utils

Module for various cluster analysis (eg extracting cluster-specific keywords)
"""

import numpy as np
import numpy.typing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction import text
from typing import Iterator, Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm
import hdbscan
import umap
from discovery_child_development import logging
from discovery_child_development.utils.openai_utils import openai
import copy


def reduce_to_2D(vectors, random_state=1):
    """Helper function to reduce vectors to 2-d embeddings using UMAP, for visualisation purposes"""
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(vectors)
    return embedding


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()

DEFAULT_STOPWORDS = stopwords.words("english")

umap_def_params = {
    "n_components": 50,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

hdbscan_def_params = {
    "min_cluster_size": 15,
    "min_samples": 1,
    "cluster_selection_method": "leaf",
    "prediction_data": True,
}


def umap_reducer(
    vectors: numpy.typing.ArrayLike,
    umap_params: dict = umap_def_params,
    random_umap_state: int = 1,
) -> numpy.typing.ArrayLike:
    """ "Reduce dimensions of the input array using UMAP"""
    logging.info(
        f"Generating {umap_def_params['n_components']}-d UMAP embbedings for {len(vectors)} vectors"
    )
    reducer = umap.UMAP(random_state=random_umap_state, **umap_params)
    return reducer.fit_transform(vectors)


def hdbscan_clustering(
    vectors: np.typing.ArrayLike, hdbscan_params: dict, have_noise_labels: bool = False
) -> np.typing.ArrayLike:
    """Cluster vectors using HDBSCAN.

    Args:
        vectors: Vectors to cluster.
        hdbscan_params: Clustering parameters.
        have_noise_labels: If True, HDBSCAN will label vectors with
            noise as -1. If False, no vectors will be labelled as noise
            but vectors with 0 probability of being assigned to any cluster
            will be labelled as -2.


    Returns:
        Dataframe with label assignment and probability of
            belonging to that cluster.
    """
    logging.info(f"Clustering {len(vectors)} vectors with HDBSCAN.")
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusterer.fit(vectors)
    if have_noise_labels:
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
    else:
        cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
        probabilities = []
        labels = []
        for probs in cluster_probs:
            probability = np.max(probs)
            label = (
                -2 if probability == 0 or np.isnan(probability) else np.argmax(probs)
            )
            probabilities.append(probability)
            labels.append(label)
        probabilities = np.array(probabilities)
        labels = np.array(labels)

    return pd.DataFrame({"labels": labels, "probability": probabilities}).astype(
        {"labels": int}
    )


def kmeans_clustering(
    vectors: np.typing.ArrayLike, kmeans_params: dict
) -> pd.DataFrame:
    """Cluster vectors using K-Means clustering"""
    logging.info(f"Clustering {len(vectors)} vectors with K-Means clustering")
    kmeans = KMeans(**kmeans_params).fit(vectors)
    return kmeans.labels_


def kmeans_param_grid_search(
    vectors: np.typing.ArrayLike, search_params: dict, random_seeds: list
) -> pd.DataFrame:
    """Perform grid search over search parameters and calculate
    mean silhouette score for K-means clustering

    Args:
        vectors: Embedding vectors.
        search_params: Dictionary with keys as parameter names
            and values as a list of parameters to search through.
        random_seeds: Param search will be performed for each
            random seed specified and then the results averaged.

    Returns:
        Dataframe with information on clustering method,
        parameters, random seed, mean silhouette score.
    """
    parameters_record = []
    silhouette_score_record = []
    method_record = []
    random_seed_record = []
    reduced_dims_vectors = umap_reducer(vectors)
    distances = euclidean_distances(reduced_dims_vectors)
    for random_seed in tqdm(random_seeds):
        for parameters in tqdm(ParameterGrid(search_params)):
            parameters["random_state"] = random_seed
            clusters = kmeans_clustering(reduced_dims_vectors, kmeans_params=parameters)
            silhouette = silhouette_score(distances, clusters, metric="precomputed")
            method_record.append("K-Means clustering")
            parameters.pop("random_state")
            parameters_record.append(str(parameters))
            silhouette_score_record.append(silhouette)
            random_seed_record.append(random_seed)

    return (
        pd.DataFrame.from_dict(
            {
                "method": method_record,
                "model_params": parameters_record,
                "random_seed": random_seed_record,
                "silhouette_score": silhouette_score_record,
            }
        )
        .groupby(["method", "model_params"])["silhouette_score"]
        .mean()
        .reset_index()
        .sort_values("silhouette_score", ascending=False)
    )


def hdbscan_param_grid_search(
    vectors: np.typing.ArrayLike, search_params: dict, have_noise_labels: bool = False
) -> pd.DataFrame:
    """Perform grid search over search parameters and calculate
    mean silhouette score for HDBSCAN

    Args:
        vectors: Embedding vectors.
        search_params: Dictionary with keys as parameter names
            and values as a list of parameters to search through.
        have_noise_labels: If True, HDBSCAN will label vectors with
            noise as -1. If False, no vectors will be labelled as noise
            but vectors with 0 probability of being assigned to any cluster
            will be labelled as -2.

    Returns:
        Dataframe with information on clustering method, parameters,
            mean silhouette scoure.
    """
    parameters_record = []
    silhouette_score_record = []
    method_record = []
    reduced_dims_vectors = umap_reducer(vectors)
    distances = euclidean_distances(reduced_dims_vectors)
    for parameters in tqdm(ParameterGrid(search_params)):
        clusters = hdbscan_clustering(
            reduced_dims_vectors,
            hdbscan_params=parameters,
            have_noise_labels=have_noise_labels,
        )
        silhouette = silhouette_score(
            distances, clusters.labels.values, metric="precomputed"
        )
        method_record.append("HDBSCAN")
        parameters_record.append(parameters)
        silhouette_score_record.append(silhouette)

    return pd.DataFrame.from_dict(
        {
            "method": method_record,
            "model_params": parameters_record,
            "silhouette_score": silhouette_score_record,
        }
    ).sort_values("silhouette_score", ascending=False)


def highest_silhouette_model_params(param_search_results: pd.DataFrame) -> dict:
    """Return dictionary of model params with the highest
    scoring mean silhouette score"""
    return param_search_results.query(
        "silhouette_score == silhouette_score.max()"
    ).model_params.values[0]


def simple_preprocessing(text: str, stopwords=DEFAULT_STOPWORDS) -> str:
    """Simple preprocessing for cluster texts"""
    text = re.sub(r"[^a-zA-Z ]+", "", text).lower()
    text = simple_tokenizer(text)
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [t for t in text if ((t not in stopwords) and (len(t) > 1))]
    return " ".join(text)


def simple_tokenizer(text: str) -> Iterator[str]:
    return [token.strip() for token in text.split(" ") if len(token) > 0]


def cluster_texts(documents: Iterator[str], cluster_labels: Iterator) -> Dict:
    """
    Creates a large text string for each cluster, by joining up the
    text strings (documents) belonging to the same cluster

    Args:
        documents: A list of text strings
        cluster_labels: A list of cluster labels, indicating the membership of the text strings

    Returns:
        A dictionary where keys are cluster labels, and values are cluster text documents
    """

    assert len(documents) == len(cluster_labels)
    doc_type = type(documents[0])

    cluster_text_dict = defaultdict(doc_type)
    for i, doc in enumerate(documents):
        if doc_type is str:
            cluster_text_dict[cluster_labels[i]] += doc + " "
        elif doc_type is list:
            cluster_text_dict[cluster_labels[i]] += doc
    return cluster_text_dict


def cluster_keywords(
    documents: Iterator[str],
    cluster_labels: Iterator[int],
    n: int = 10,
    tokenizer=simple_tokenizer,
    max_df: float = 0.90,
    min_df: float = 0.01,
    Vectorizer=TfidfVectorizer,
    ngram_range: Tuple[int, int] = (1, 1),
) -> Dict:
    """
    Generates keywords that characterise the cluster, using the specified Vectorizer

    Args:
        documents: List of (preprocessed) text documents
        cluster_labels: List of integer cluster labels
        n: Number of top keywords to return
        Vectorizer: Vectorizer object to use (eg, TfidfVectorizer, CountVectorizer)
        tokenizer: Function to use to tokenise the input documents; by default splits the document into words

    Returns:
        Dictionary that maps cluster integer labels to a list of keywords
    """
    my_stop_words = text.ENGLISH_STOP_WORDS

    # Define vectorizer
    vectorizer = Vectorizer(
        analyzer="word",
        tokenizer=tokenizer,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_df=max_df,
        min_df=min_df,
        max_features=10000,
        stop_words=list(my_stop_words),
        ngram_range=ngram_range,
    )

    # Create cluster text documents
    cluster_documents = cluster_texts(documents, cluster_labels)
    unique_cluster_labels = list(cluster_documents.keys())

    # Apply the vectorizer
    token_score_matrix = vectorizer.fit_transform(list(cluster_documents.values()))

    # Create a token lookup dictionary
    id_to_token = dict(
        zip(list(vectorizer.vocabulary_.values()), list(vectorizer.vocabulary_.keys()))
    )

    # For each cluster, check the top n tokens
    top_cluster_tokens = {}
    for i in range(token_score_matrix.shape[0]):
        # Get the cluster feature vector
        x = token_score_matrix[i, :].todense()
        # Find the indices of the top n tokens
        x = list(np.flip(np.argsort(np.array(x)))[0])[0:n]
        # Find the tokens corresponding to the top n indices
        top_cluster_tokens[unique_cluster_labels[i]] = [id_to_token[j] for j in x]

    return top_cluster_tokens


def get_cluster_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> List[np.ndarray]:
    """Get the centroids of each cluster

    Args:
        data_df: Dataframe with integer index [0, 1, 2...] and a column 'cluster' with cluster labels
        embeddings: Embeddings for each data point

    Returns:
        List of cluster centroid vectors
    """
    centroids = []
    for i in range(len(df["cluster"].unique())):
        cluster = df[df["cluster"] == i]
        centroid = np.mean(embeddings[cluster.index], axis=0)
        centroids.append(centroid)
    return centroids


def get_n_most_central_vectors(
    embeddings: np.ndarray, centroid: np.ndarray, n: int = 10
) -> np.ndarray:
    """Get the n most similar data points to a centroid"""
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argsort(distances)[:n]


def describe_clusters_with_gpt(
    cluster_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_central: int = 15,
    gpt_message: str = None,
) -> List[str]:
    """
    Generate cluster descriptions from cluster centroids using GPT

    Args:
        cluster_df (pd.DataFrame): Dataframe with integer index [0, 1, 2...] and columns
            'cluster' with cluster labels, and 'text' with text data for each data point
        gpt_message (str): Message to send to GPT to generate cluster description
        embeddings (np.ndarray): Embeddings for each data point used for determining cluster centroids
        n_central (int, optional): Number of most central data points to use for cluster description. Defaults to 15.

    Returns:
        List[str]: List of cluster descriptions
    """
    if gpt_message is None:
        gpt_message = "Here are the most central documents of a document cluster. \
            Describe what kind of information is this cluster capturing, in 2 sentences. \
            \n\n##Abstracts\n\n {} \n\n##Description (2 short sentences)"

    # Get cluster centroid indices
    centroids = get_cluster_centroids(cluster_df, embeddings)
    most_central = []
    for i in range(len(centroids)):
        most_central.append(
            get_n_most_central_vectors(embeddings, centroids[i], n=n_central)
        )

    # Generate cluster descriptions
    cluster_descriptions = []
    for i in range(len(centroids)):
        abstracts = cluster_df.iloc[most_central[i]].text.to_list()
        messages = [
            {
                "role": "user",
                "content": copy.deepcopy(gpt_message).format("\n".join(abstracts)),
            }
        ]
        chatgpt_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6,
            max_tokens=1000,
        ).to_dict()
        cluster_descriptions.append(chatgpt_output["choices"][0]["message"]["content"])

    return cluster_descriptions


def generate_cluster_names_with_gpt(
    cluster_descriptions: List[str], gpt_message: str = None
) -> Dict[int, str]:
    """Generate cluster names from cluster descriptions using GPT

    Args:
        cluster_descriptions (List[str]): List of cluster descriptions
        gpt_message (str): Message to send to GPT to generate cluster names

    Returns:
        Dict[int, str]: Dictionary mapping cluster index to cluster name
    """
    # Prepare the GPT message
    cluster_descriptions_with_numbers = [
        f"{i}: {x}" for i, x in enumerate(cluster_descriptions)
    ]
    if gpt_message is None:
        gpt_message = "Summarise these cluster descriptions in 2-3 words.\n\n##Descriptions\n\n {}"
    messages = [
        {"role": "user", "content": gpt_message.format("\n".join(cluster_descriptions))}
    ]
    # Generate cluster names
    cluster_names = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.6,
        max_tokens=1000,
    ).to_dict()
    # Get cluster names from the GPT response
    cluster_names_ = cluster_names["choices"][0]["message"]["content"].split("\n")
    # Map cluster index to cluster name
    return {i: cluster_name for i, cluster_name in enumerate(cluster_names_)}
