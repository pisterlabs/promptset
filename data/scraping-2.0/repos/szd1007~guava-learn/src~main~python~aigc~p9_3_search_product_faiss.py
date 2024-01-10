import faiss
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd

embedding_model = "text-embedding-ada-002"

def load_embeddings_to_faiss(df):
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(index, df, query, k=5):
    query_vector = np.array(get_embedding(query, engine=embedding_model)).reshape(1, -1).astype('float32')
    distances, indexes = index.search(query_vector, k)

    results = []
    for i in range(len(indexes)):
        product_names = df.iloc[indexes[i]]['product_name'].values.tolist()
        results.append((distances[i], product_names))
    return results

df = pd.read_parquet("/Users/zm/aigcData/my_taobao_produtct_title.parquet")
index = load_embeddings_to_faiss(df)
products = search_index(index, df, "自然淡雅背包", k=3)

for distances, product_names in products:
    for i in range(len(distances)):
        print(product_names[i], distances[i])

