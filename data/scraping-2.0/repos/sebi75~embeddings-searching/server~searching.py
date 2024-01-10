from openai.embeddings_utils import cosine_similarity
import numpy as np
import utils


def search(df, searchTerm, n=3, pprint=True):
    df['embedding'] = df.embedding.apply(
        eval).apply(np.array)

    queryEmbedding = utils.get_embedding(
        searchTerm, model='text-embedding-ada-002')
    df['similarities'] = df.embedding.apply(
        lambda x: cosine_similarity(x, queryEmbedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res
