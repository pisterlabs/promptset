import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
#import constants

def pytest_configure(config):
    #os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
    #os.environ["PINECONE_API_KEY"] = constants.PINECONE_API_KEY
    #os.environ["MONGODB_URI"] = constants.MONGODB_URI
    #os.environ["DB_URI"] = constants.DB_URI
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    openai.api_key = os.environ["OPENAI_API_KEY"]
    pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment="northamerica-northeast1-gcp"
    )