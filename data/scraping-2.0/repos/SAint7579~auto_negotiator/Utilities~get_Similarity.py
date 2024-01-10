import openai
import chromadb
from chromadb.utils import embedding_functions
import json
import pandas as pd
vendor = pd.read_csv("C:/Users/vishw/OneDrive/Desktop/Projects/auto_negotiator/Utilities/vendors.csv")

with open('C:/Users/vishw/OneDrive/Desktop/Projects/daemon-dialoguers/openAI_api.json') as f:
    key = json.load(f)

def text_Embedding(text):
    response = openai.OpenAI(api_key='')
    response = response.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

def get_Similarity(query):
    client = chromadb.Client()
    collection = client.get_or_create_collection("vendor",embedding_function=openai_ef)
    
    docs=vendor["description"].tolist() 
    ids= [str(x) for x in vendor.index.tolist()]
    
    collection.add(
    documents=docs,
    ids=ids
    )
    
    vector=text_Embedding(query)
    
    results=collection.query(    
    query_embeddings=vector,
    n_results=5,
    include=["documents"])
    lis=[]
    for i in results['ids'][0]:
        lis.append(vendor.iloc[(int(i))]["id"])

    return lis