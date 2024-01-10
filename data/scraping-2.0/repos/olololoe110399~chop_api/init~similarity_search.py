import json
from typing import List

import numpy as np
import openai
import pandas as pd


class SimilaritySearch:
    def __init__(self, json_path_file):
        self.client = openai.Client()
        with open(json_path_file, 'r') as file:
            self.data = json.load(file)
        self.df = pd.DataFrame(self.data)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embbeding(self, text, model='text-embedding-ada-002'):
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def similarity_search(self, user_query: List[str], top_n=3):
        results = []

        for query in user_query:
            embedding = self.get_embbeding(query, model='text-embedding-ada-002')
            self.df['similarities'] = self.df.embeddings.apply(lambda x: self.cosine_similarity(x, embedding))
            res = self.df.sort_values(by='similarities', ascending=False).head(top_n)
            results.append(res)
        return results
