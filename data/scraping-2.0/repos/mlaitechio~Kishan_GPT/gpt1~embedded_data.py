import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") 
openai.api_key = API_KEY
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

df = pd.read_csv('ICICI_all_data3.csv')
# count length of content field in csv
df["token"] = None
for idx, r in df.iterrows():
#     print(len(r.content))
#     df["token"] = df[len(r.content)]
    df.loc[idx,'token'] = len(r.content)
   
# df = pd.DataFrame.from_dict(df)
# df.to_csv("icici_with_token.csv") #? give this csv to embedd

# embedd the given csv data 
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    """embedd the given csv data """
    ccontent = text.encode(encoding='ASCII',errors='ignore').decode() #fix any UNICODE error
    
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

import time
# embedd particular column of csv in openai
def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """ embedd particular column of csv in openai"""
    embeddings = {}
    for idx, r in df.iterrows():
        print(idx)
#         print(r)
        embeddings[idx] = get_embedding(r.title)
        time.sleep(5)  # Add a delay of 10 second between requests
    return embeddings

document_embeddings = compute_doc_embeddings(df)

#? Make CSV of EMBEDDEd data
# df2 = pd.DataFrame.from_dict(document_embeddings, orient='index')

# df2.index.names = ['title']
# df2.columns = [str(i) for i in range(df2.shape[1])]
# # columns = [len(values) for i in ranfe(df.shape[1])]
# # print(columns)
# # Replace `filename.csv` with the desired filename/path
# df2.to_csv('icici_embed.csv')


 
 