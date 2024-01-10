import openai
import numpy as np


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    return np.array(response)