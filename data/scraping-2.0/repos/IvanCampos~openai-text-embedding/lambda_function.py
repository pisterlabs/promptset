import os
import io
import openai
import numpy as np
from numpy.linalg import norm
import pandas as pd

def lambda_handler(event, context):

    openai.api_key = os.getenv('OPENAI_API_KEY')

    df = pd.read_csv('words.csv')
    # print(df)

    input_term = "the fox crossed the road"
    input_term_embeddings = get_embeddings_for_text(input_term)

    # print(input_term_embeddings)

    df['embedding'] = df['text'].apply(lambda x:get_embeddings_for_text(x))

    output = df.to_csv(index=False)
    df = pd.read_csv(io.StringIO(output))
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    # print(df)

    search_term = "dunkin"

    search_term_vector_embeddings = get_embeddings_for_text(search_term)

    # print(search_term_vector_embeddings)

    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector_embeddings))

    df_top = df.sort_values("similarities", ascending=False).head(10)

    df_return = df_top[['text', 'similarities']].to_json()

    # print(df)

    return {
        'statusCode': 200,
        'body': df_return
    }

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def get_embeddings_for_text(input_term):
    input_vector = openai.Embedding.create(
        input = input_term,
        model="text-embedding-ada-002")
    input_vector_embeddings = input_vector['data'][0]['embedding']
    return input_vector_embeddings