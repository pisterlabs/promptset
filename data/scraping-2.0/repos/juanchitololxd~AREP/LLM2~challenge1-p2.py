import os
from langchain.embeddings import OpenAIEmbeddings

def buscar():
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter")
    embeddings = OpenAIEmbeddings(openai_api_key="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")



