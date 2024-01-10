import openai
import numpy as np
from transformers import GPT2TokenizerFast
import pickle

EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
COMPLETION_MODEL_NAME = "text-davinci-003"

SEPARATOR = "\n"
MAX_CONTEXT_LEN = 2048

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def get_embedding(text):
    result = openai.Embedding.create(
        model=EMBEDDINGS_MODEL_NAME,
        input=text
    )
    return result["data"][0]["embedding"]


def get_similarities(query_embedding, embeddings):
    return sorted([
        (vector_similarity(query_embedding, embedding), text, tokens)
        for text, embedding, tokens
        in embeddings], reverse=True)


def load_embeddings(embeddings_file):
    with open(embeddings_file, "rb") as f:
        return pickle.load(f)


def get_context(similiarities):
    context = []
    context_len = 0

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    for _, text, tokens in similiarities:
        context_len += tokens + separator_len
        if context_len > MAX_CONTEXT_LEN:
            if len(context) == 0:
                context.append(text[:(MAX_CONTEXT_LEN - separator_len)])
            break
        context.append(text)
    return context

def get_completion(prompt):
    response = openai.Completion.create(
        model=COMPLETION_MODEL_NAME,
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

def ask(query, embeddings_file):
    embeddings = load_embeddings(embeddings_file)
    query_embedding = get_embedding(query)
    similiarities = get_similarities(query_embedding, embeddings)
    context = get_context(similiarities)

    # Borrowed from https://github.com/openai/openai-cookbook/blob/838f000935d9df03e75e181cbcea2e306850794b/examples/Question_answering_using_embeddings.ipynb
    prompt = f"Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say \"I don't know.\"\n\nContext:\n{context} \n\nQuestion:\n{query}\n\nAnswer:"

    return get_completion(prompt)
