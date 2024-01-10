import os.path

import openai
from typing import List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from openai.embeddings_utils import get_embedding
import tiktoken
import numpy as np
class EmbeddingManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        # embedding model parameters
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
        self.max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

    def load(self) -> pd.DataFrame:
        '''
        Will load the dataset
        :return:
        '''
        # load & inspect dataset
        input_datapath = "data/fine_food_reviews_1k.csv"
        df = pd.read_csv(input_datapath, index_col=0)
        df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
        df = df.dropna()
        df["combined"] = (
                "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
        )
        df.head(2)

        # subsample to 1k most recent reviews and remove samples that are too long
        top_n = 1000
        df = df.sort_values("Time").tail(
            top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
        df.drop("Time", axis=1, inplace=True)

        encoding = tiktoken.get_encoding(self.embedding_encoding)

        # omit reviews that are too long to embed
        df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
        df = df[df.n_tokens <= self.max_tokens].tail(top_n)
        len(df)

        return df

    def get_2d_pca_projection(self,df: pd.DataFrame):
        '''
        Will return 2d PCA projection of the given question embeddings
        :param texts:
        :return:
        '''
        #project embeddings to 2d
        # Create a t-SNE model and transform the data
        matrix = df.embedding.apply(eval)

        #convert list of vectors to numpy array
        matrix = np.array(matrix.tolist())

        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random',
                    learning_rate=200)
        vis_dims = tsne.fit_transform(matrix)

        colors = ["red", "darkorange", "gold", "blue", "darkgreen"]
        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]
        color_indices = df.Score.values - 1

        colormap = matplotlib.colors.ListedColormap(colors)
        plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
        #add legend for colors to plot
        handles = [matplotlib.patches.Patch(color=colors[i], label=f"{i+1}") for i in range(5)]
        plt.legend(handles=handles)
        #add legend title
        plt.gca().get_legend().set_title("Amazon rating")
        plt.title("Amazon ratings visualized in language using t-SNE")
        plt.show()

    def cluster_texts(self, texts: List[List[float]],k=2) -> List[int]:
        '''
        Will cluster questions using the K-NN algorithm and return cluster ids for each question embedding
        Args:
            question_embeddings: List of question embeddings (each embedding is a list of floats)

        Returns: List of cluster ids for each question embedding
        '''

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(texts)
        #return cluster id for each question
        return kmeans.labels_

    def embed_batch(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        Will embed a batch of texts
        :param texts:
        :return:
        '''
        # Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

        # This may take a few minutes
        df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=self.embedding_model))
        df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")
        return df

if __name__ == "__main__":
    embeddings_manager = EmbeddingManager()
    #load texts
    df = embeddings_manager.load()
    #embed texts
    if os.path.exists("data/fine_food_reviews_with_embeddings_1k.csv"):
        print("Loading embeddings from file")
        df = pd.read_csv("data/fine_food_reviews_with_embeddings_1k.csv", index_col=0)
        embeddings = df.embedding.values.tolist()
        #convert to float from string
        embeddings = [eval(x) for x in embeddings]
    else:
        embeddings = embeddings_manager.embed_batch(df=df)
    #cluster texts
    cluster_ids = embeddings_manager.cluster_texts(texts=embeddings)
    print("Cluster ids:")
    print(cluster_ids[0:5])
    #visualize embeddings
    embeddings_manager.get_2d_pca_projection(df=df)
