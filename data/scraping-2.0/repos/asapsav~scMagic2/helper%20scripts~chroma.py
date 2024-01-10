import streamlit as st
import nbformat
from io import BytesIO
import tiktoken
import dotenv
import os
import openai
from nbconvert import HTMLExporter
import chromadb
import pandas as pd

from prompts import PLANNER_AGENT_MINDSET

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#import df with tools and scraped readmes
current_directory = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_directory, "..", 'datasets', 'tool-table-with-readmes.csv')
df = pd.read_csv(file_path)

# embed vectors
chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.
# Uncomment for persistent client
# chroma_client = chromadb.PersistentClient()
EMBEDDING_MODEL = "text-embedding-ada-002"
# change this to biotech specialised model later
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBEDDING_MODEL)
scrnatools_description_collection = chroma_client.create_collection(name='scRNA_Tools_2', embedding_function=embedding_function)
scrnatools_description_collection.add(
    documents = list(df['extented_desc_readme_trim']),
    metadatas = df.drop(['extented_desc_readme_trim'], axis = 1).to_dict(orient='records'),
    ids = list(df.Name))

# Query DB
def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
                'id':results['ids'][0],
                'score':results['distances'][0],
                'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc_readme_trim'],
                'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
                })

    return df['content'].tolist() , df['id'].tolist()


query_collection(scrnatools_description_collection, 'quality controll python', 3, df)