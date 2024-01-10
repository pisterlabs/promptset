import os
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv("../.env")

df = pd.read_csv('data/taaft.tsv', sep='\t')

# Instance of ChromaDB collection for AI tools
langchain_chroma = Chroma(
    collection_name="taaft",
    client=chromadb.PersistentClient(path="./taaft-db"),
    embedding_function=OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
)

# Find the most relevant AI tools using similarity search on vector embeddings
def predict(query):
    if len(query) < 100:
        return []
    docs = langchain_chroma.similarity_search(query, k=5)    
    results = list(map( 
      lambda x: {
        "title": df["title"][int(x.metadata["index"])],
        "description": x.page_content[1:-1],
        "link": df["link"][int(x.metadata["index"])]
      },
      filter(lambda x: not x.page_content.startswith("|Unfortunately"), docs)
    ))
    
    return results