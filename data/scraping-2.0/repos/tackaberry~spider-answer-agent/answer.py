
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['default']['openaiApiKey']

def createContext( df, question ):

    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    max_len = 2000

    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)

def createPrompt( context, question):
    return f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"

def completePrompt(prompt):

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0.1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=config['default']['model'],
    )
    return response["choices"][0]["text"].strip()

def getAnswer( question): 

    df=pd.read_csv('data/processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()

    context = createContext( df, question )
    prompt = createPrompt( context, question )
    response = completePrompt(prompt)

    return response

