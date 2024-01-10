import os
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
import pickle

import dotenv
dotenv.load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def join_data(labels):
    return pd.concat([pd.read_excel(label) for label in labels])

def df_add_2darray(df, matrix, col_name, delimiter='|'):
    df = df.copy()
    str = '\n'.join(delimiter.join('%f' %x for x in y) for y in matrix)
    row = str.split("\n")
    df.loc[:, col_name] = row
    return df

def df_extract_2darray(df, col_name, delimiter='|'):
    arr = df[col_name].tolist()
    matrix = [np.array([float(x) for x in row.split(delimiter)]) for row in arr]
    return np.array(matrix)

def get_embedding(text):
    embedding = openai.Embedding.create(input=[text], model="text-embedding-ada-002")['data'][0]['embedding']
    return embedding

def get_embeddings(df, target):
    targets = df[target].tolist()
    keywords = np.array(targets)
    matrix = np.array([get_embedding(keyword) for keyword in tqdm(keywords)])
    return matrix

def save_embeddings(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"Saved embeddings to {filename}")

def load_embedding(filename):
    with open(filename, 'rb') as f:
        matrix = pickle.load(f)
    return matrix

def load_embeddings(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    matrix = np.concatenate(tuple([load_embedding(filename) for filename in filenames]), axis=0)
    return matrix

def filter_df_and_embeddings(df, subset, embeddings, column="tweet type"):
    df = df.reset_index()
    if subset:
        indices = df[df[column] == subset].index.tolist()
    filtered_df = df.iloc[indices] if subset else df
    filtered_embeddings = embeddings[indices] if subset else embeddings
    return filtered_df, filtered_embeddings

