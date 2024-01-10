import os
from pprint import pprint

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
    cosine_similarity
)

load_dotenv()

openai.api_key = os.getenv('OPEN_AI_API_KEY')


def search_reviews(df, query, n=3):
    embedding = get_embedding(query, engine='text-search-curie-query-001')
    df['similarities'] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res['0'].to_list()


def recommendations_from_strings(df, query, n=3):
    embedding = get_embedding(query, engine='text-similarity-curie-001')
    distances = distances_from_embeddings(embedding, df.curie_similarity, distance_metric="cosine")
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    result = []
    for i in indices_of_nearest_neighbors[:n]:
        result.append(df['0'][i])

    return result


def main():
    df = pd.read_csv('wine_tasting_notes_embeddings__curie_combined.csv')
    df['curie_similarity'] = df.curie_similarity.apply(eval).apply(np.array)
    df['curie_search'] = df.curie_search.apply(eval).apply(np.array)

    search = search_reviews(df, 'Red wine with hints of chestnut, fig and bitter-orange', n=5)
    pprint(search)

    recommend = recommendations_from_strings(df, 'Red wine with hints of chestnut, fig and bitter-orange', n=5)
    pprint(recommend)


if __name__ == '__main__':
    main()
