import pickle

import openai
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, TSNE
from umap import UMAP


def calculate_2d_embeddings(df: pd.DataFrame, perplexity=5, random_state=42):
    """This function plots the t-SNE embeddings of the long text.
    Args:
        df (pd.DataFrame): The dataframe containing the text.
        perplexity (int): The perplexity to use for the t-SNE. Defaults to
        5.
        random_state (int): The random state to use for the t-SNE.
        Defaults to 42.
    Returns:
        fig (matplotlib.pyplot.figure): The figure of the t-SNE plot.
    """

    # Start by calculating embeddings
    papers = (df.title.fillna("") + "\n" + df.abstract.fillna("")).values
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(papers, show_progress_bar=True)

    # Create a t-SNE model and transform the data
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate=200,
    )
    vis_dims = model.fit_transform(embeddings)
    df["tsne_x"] = vis_dims[:, 0]
    df["tsne_y"] = vis_dims[:, 1]

    # Create a MDS model and transform the data
    model = MDS(
        n_components=2,
        random_state=random_state,
    )
    vis_dims = model.fit_transform(embeddings)
    df["mds_x"] = vis_dims[:, 0]
    df["mds_y"] = vis_dims[:, 1]

    # Create a MDS model and transform the data
    model = UMAP(
        n_components=2,
        random_state=random_state,
    )
    vis_dims = model.fit_transform(embeddings)
    df["umap_x"] = vis_dims[:, 0]
    df["umap_y"] = vis_dims[:, 1]

    return df


def request_label(titles):
    titles = "\n".join(titles)
    prompt = f"Give a 2-5 word label that summarizes the common topic of these abstracts. Avoid vague labels like 'artificial intelligence', 'machine learning', 'neuroscience' and 'deep learning'\n\n {titles}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion["choices"][0]["message"]["content"].split(".")[0]


def main():
    df = pd.read_csv("data/processed/neuroai-works.csv")
    df.sort_values("ss_cited_by_count", ascending=False, inplace=True)

    df = calculate_2d_embeddings(df)
    df.to_csv("data/processed/neuroai-works-umap.csv", index=False)

    df = df[df["openai_category"].isin(["A", "B", "C"])]

    kmeans = KMeans(n_clusters=25, random_state=0, n_init="auto").fit(
        df[["umap_x", "umap_y"]]
    )
    labels = kmeans.fit_predict(df[["umap_x", "umap_y"]].values)

    label_map = []
    for label in tqdm.tqdm(range(kmeans.cluster_centers_.shape[0])):
        titles = df.iloc[labels == label].title.values.tolist()
        label_name = request_label(titles)
        label_map.append(label_name)

    label_info = {
        "label_centers": kmeans.cluster_centers_,
        "labels": label_map,
        "paper_labels": [label_map[x] for x in labels],
    }

    with open("data/processed/umap_labels.pickle", "wb") as f:
        pickle.dump(label_info, f)


if __name__ == "__main__":
    main()
