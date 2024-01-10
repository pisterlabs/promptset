# Description: This file creates a Pinecone index for the keyword search


import pinecone
import os
import openai
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("Please set your PINECONE_API_KEY environment variable")

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone.create_index("keyword-search", dimension=1536, metric="dotproduct")
