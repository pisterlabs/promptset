# imports
import ast
from math import cos  # for converting embeddings saved as strings back to arrays
from openai import AzureOpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv
import numpy as np

EMBEDDING_MODEL = "textembedding"
def query():
    # models
    EMBEDDING_MODEL = "textembedding"
    GPT_MODEL = "gpt-3.5-turbo"

    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-10-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    deployment_name='GPT4'
    deployment_name="gpt35turbo"

    # an example question about the 2022 Olympics
    query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

    response = client.chat.completions.create(
        model = deployment_name,
        messages=[
            {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},        
            {"role": "user", "content": query}
        ]
    )
    #print(response.model_dump_json(indent=2))
    print(response.choices[0].message.content)

def load_data():
    embeddings_path = "winter_olympics_2022.csv"
    df = pd.read_csv(embeddings_path)
    
    # convert embeddings from CSV str type back to list type
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df

def pddata():
    embeddings_path = "winter_olympics_2022.csv"
    df = pd.read_csv(embeddings_path)
    #print(df)
    #for i in range(10):
    #    print(df.iloc[i].loc["embedding"])
    print("########")
    print(df.iloc[3].loc["embedding"])
    print("########")
    # convert embeddings from CSV str type back to list type
    #df['embedding'] = df['embedding'].apply(ast.literal_eval)
    print("--------")
    print(df.iloc[3].loc["embedding"])
    print("===========")
    print(df["text"][100])
    print("===========")
    print(df["embedding"][100])

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def generate_embeddings(text, model="textembedding"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    #df = load_data()
    load_dotenv()
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = "2023-05-15",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    #pddata()

    df = load_data()
    emb1 = df["embedding"][100]
    text = df["text"][100]
    print("===***********")
    emb2 = client.embeddings.create(input = [text], model=EMBEDDING_MODEL).data[0].embedding
    print(emb2)
    similarity = cosine_similarity(emb1,emb2)
    print(f"simililarity : {similarity}")
    #df_bills['ada_v2'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = 'text-embedding-ada-002')) 