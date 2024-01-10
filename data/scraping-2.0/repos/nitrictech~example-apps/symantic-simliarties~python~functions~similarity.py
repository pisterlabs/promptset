import os
import openai
import numpy as np
import pandas as pd
from nitric.resources import api
from nitric.application import Nitric
from numpy.linalg import norm

publicApi = api("main")
openai.api_key = os.getenv('OPENAI_SECRET_KEY')
openai_model = os.getenv('OPENAI_MODEL')


@publicApi.post("/similarity")
async def similarity(ctx):
    data = pd.read_csv('dictionary.csv')
    data['embedding'] = data['text'].apply(generate_embedding)

    search_term = ctx.req.json['search_term']
    search_embedding = generate_embedding(search_term)

    data["similarities"] = data['embedding'].apply(
        lambda x: cosine_similarity(x, search_embedding))
    sorted_data = data.sort_values("similarities", ascending=False).head(5)

    ctx.res.body = sorted_data[['text', 'similarities']].to_json()


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def generate_embedding(text):
    input_vector = openai.Embedding.create(input=text, model=openai_model)
    return np.array(input_vector['data'][0]['embedding'])


Nitric.run()
