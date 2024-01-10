'''from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from .config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV

def get_docsearch_from_existing():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    index_name = 'texttutor-b0620a2025aa4ceea3047863bf9a56f1'

   
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch'''
