import pandas as pd
import numpy as np
from ast import literal_eval
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = "<API_KEY>"

datafile_path = "Data_embedded.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
def search_reviews(df, reviewDescription, n=3, pprint=True):
    product_embedding = get_embedding(
        reviewDescription,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "Good Product", n=3)