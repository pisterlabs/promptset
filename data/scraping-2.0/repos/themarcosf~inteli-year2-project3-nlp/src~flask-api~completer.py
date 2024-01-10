import pandas as pd
import openai
from ast import literal_eval
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import os


def create_context(question):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    openai.api_key = "sk-..."

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, model='text-embedding-ada-002')['data'][0]['embedding']

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    print(df.head())

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > 1800:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def complete(q):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(q)

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Responda a questão com base no context abaixo, e se a questão não puder ser respondida com base no contexto, diga \"Eu não tenho informações para responder a questão\"\n\nContexto: {context}\n\n---\n\nQuestão: {q}\nResposta:",
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            model="text-davinci-003",
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
