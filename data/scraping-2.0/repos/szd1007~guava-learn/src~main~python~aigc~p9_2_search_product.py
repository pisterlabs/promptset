from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd

embedding_model = "text-embedding-ada-002"


# search through the reviews for a specific product
def search_product(df, query, n=3, pprint=True):
    product_embedding = get_embedding(
        query,
        engine=embedding_model
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .product_name
    )
    if pprint:
        for r in results:
            print(r)
    return results


df = pd.read_parquet("/Users/zm/aigcData/my_taobao_produtct_title.parquet")

results = search_product(df, "小米", n=3)