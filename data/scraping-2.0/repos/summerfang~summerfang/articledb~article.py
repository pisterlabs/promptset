import os
from ast import literal_eval

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

from openai import OpenAI

def distances_from_embeddings(query_embedding, embeddings):
    """
    Calculate the cosine similarity between each embedding in `embeddings` and `query_embedding`.

    Args:
        embeddings (List[List[float]]): A list of embeddings, where each embedding is a list of floats.
        query_embedding (List[float]): The query embedding, represented as a list of floats.

    Returns:
        List[float]: A list of cosine similarities between each embedding in `embeddings` and `query_embedding`.
    """
    return [1 - cosine(embedding, query_embedding) for embedding in embeddings]


def create_context(
    client, question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    # q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002')['data'][0]['embedding']
    # q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002')
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values)

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    # df=df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    try:    
        client = OpenAI()

    except Exception as e:
        print(e)
        return "openai client error"
    
    df = pd.read_csv('./articledb/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

    context = create_context(
        client,
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                {"role": "user", f"content": "Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return "I don't know"
    
# answer = answer_question(question="How about reuse the code?")
# print(answer)