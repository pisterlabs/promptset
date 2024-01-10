import os
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from google.cloud import storage
import io
from utils.embedding_utils import get_embedding
from tenacity import retry, stop_after_attempt, wait_random_exponential

n_relevant_chunks = 3
storage_client = storage.Client()

with open("./data/db/system_prompt.txt") as f:
    """Load system prompt."""
    system_prompt = " ".join(line.rstrip() for line in f)

with open("./data/db/context_prompt.txt") as f:
    """Load context prompt."""
    context_prompt = " ".join(line.rstrip() for line in f)


def semantic_search(query_embedding, embeddings):
    """Indices of embeddings ranked by similarity to query embedding."""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    ranked_indices = np.argsort(-similarities)
    return ranked_indices


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """Chat completion with exponential backoff to prevent rate limiting."""
    return openai.ChatCompletion.create(**kwargs)


def answer_question(chunk, question, api_key=None, model="gpt-3.5-turbo", max_tokens=300, temperature=0.25):
    """Return answer to query given context chunk."""

    if api_key is None:  # Use system API key
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    else:  # Use user API key
        openai.api_key = api_key

    prompt = f"Use the following context to answer the question at the end.\nContext: {chunk}.\n{context_prompt}\nQuestion: {question}"
    response = completion_with_backoff(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


# Cache data in memory to avoid loading from GCS on every request
data_cache = {}


def load_csv_file_from_gcs(bucket_name, file_name):
    """Load csv file from GCS bucket into memory and return as Pandas DataFrame."""
    if file_name in data_cache:
        return data_cache[file_name]
    gcs_path = f"gs://{bucket_name}/{file_name}"
    data = pd.read_csv(gcs_path)
    data_cache[file_name] = data
    return data


def load_npy_file_from_gcs(bucket_name, file_name):
    """Load npy file from GCS bucket into memory and return as NumPy array."""
    if file_name in data_cache:
        return data_cache[file_name]
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    npy_string = blob.download_as_bytes()
    npy_file = io.BytesIO(npy_string)
    data = np.load(npy_file)
    data_cache[file_name] = data
    return data


def run(query, api_key=None):
    """Run ChatJesseT on query and return an answer."""
    if not query:
        return "Please enter your question above, and I'll do my best to help you."
    if len(query) > 150:
        return "Please ask a shorter question!"
    else:
        # Load remote files
        embeddings = load_npy_file_from_gcs("chatjesset.appspot.com", "embeddings.npy")
        df_text = load_csv_file_from_gcs("chatjesset.appspot.com", "text_chunks.csv")

        # # Load local files
        # df_text = pd.read_csv("data/db/text_chunks.csv")
        # embeddings = np.load("data/db/embeddings.npy")

        query_embedding = get_embedding(query, api_key=api_key)
        ranked_indices = semantic_search(np.array(query_embedding), embeddings)
        most_relevant_chunk = " ".join(df_text.loc[ranked_indices[:n_relevant_chunks], "text_chunks"].values.flatten())
        answer = answer_question(most_relevant_chunk, query, api_key)
        answer.strip("\n")
        return answer
