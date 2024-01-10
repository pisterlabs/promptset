import os

import openai
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.vectorstores import Pinecone
import pinecone

from service.get_langchain_service import set_langchain_key
from utils.utils import set_openai_key, VectorDB


class PineconeService:
    def __init__(self, index_name="vectordatabase"):
        self.vectordb = VectorDB(index_name)

    def query_match(self, input_text):
        self.vectordb.query_match(input_text)

if __name__ == "__main__":
    pinecone_service = PineconeService()
    pinecone_service.vectordb.list_indexes()
    pinecone_service.query_match("what is diabetes?")



