from dotenv import load_dotenv
load_dotenv()
import os
import openai
import pinecone
from langchain import OpenAIEmbeddings
#initialize pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_API_ENV']  # next to api key in console
)
index_name = "fast-and-slow"   

if pinecone.list_indexes():
    pinecone.delete_index(pinecone.list_indexes()[0])


#initialize our embeddings and openai related things
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
openai.api_key = os.environ['OPENAI_API_KEY']