"""
Search using OpenAI
"""
import openai
import pandas as pd
import time
import streamlit as st
from openai.embeddings_utils import cosine_similarity

RESULT_MIN_SCORE = 0.5  # don't inlcude content from result below this score

TOP_K = 16  # Number of results we want to retrieve

MODEL_NAME = "text-embedding-ada-002"  # only used for logging

def search(query: str, top_k: int = TOP_K) -> list({int, float}):
    start_time = time.time()

    embedding_response = get_single_embedding(query)
    # embedding_response = MOCK_OAI_RESPONSE = {
    #     "data": [{"embedding": df['embeddings'][0]}], "usage": {"total_tokens": 0}}

    question_embedding = embedding_response["data"][0]["embedding"]

    token_usage = embedding_response["usage"]["total_tokens"]

    hits = df.apply(
        lambda row: cosine_similarity(row["embeddings"], question_embedding), axis=1)

    hits = hits.sort_values(ascending=False).head(top_k)

    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query, "total_tokens:", token_usage)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))

    hits = hits.reset_index(name='score')
    hits = hits.rename(columns={'index': 'corpus_id'})

    hits = hits.to_dict("records")

    return hits


@st.cache_data
def get_corpus() -> pd.DataFrame:
    """ Returns the corpus loaded as dataframe """
    return df


@st.cache_data
def get_single_embedding(text: str) -> object:
    """
    Get a single embedding from OpenAI API 
    returns the full response object (token usage data)
    """
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )
    return response

df = pd.read_parquet('corpus/embeddings_oai_ada.parquet')
print("Corpus loaded from corpus/embeddings_oai_ada.parquet")
