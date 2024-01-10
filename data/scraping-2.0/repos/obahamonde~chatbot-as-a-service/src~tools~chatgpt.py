from os import environ
from typing import *

import pinecone as pc
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores.pinecone import Pinecone
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

load_dotenv()

OPENAI_API_KEY:str = environ["OPENAI_API_KEY"]
OPENAI_BASE_URL:str = environ["OPENAI_BASE_URL"]
PINECONE_API_KEY:str = environ["PINECONE_API_KEY"]
PINECONE_INDEX:str = environ["PINECONE_INDEX"]
PINECONE_API_URL:str = environ["PINECONE_API_URL"]
PINECONE_ENVIRONMENT:str = environ["PINECONE_ENVIRONMENT"]

pc.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
openai_embeddings = OpenAIEmbeddings(client=None)
openai_chat = ChatOpenAI(client=None)
pinecone = Pinecone.from_existing_index(embedding=openai_embeddings, index_name=PINECONE_INDEX)


class ChatGPT(BaseModel):
    namespace:str = Field(..., description="The namespace to store the embeddings")
    top_k:int = Field(default=4, description="The number of results to return")
    
    async def insert(self, documents:List[str]):
        """Inserts a list of documents into the vector store"""
        await pinecone.aadd_texts(documents, namespace=self.namespace)
    async def search(self, query:str):
        """Searches the vector store for the query"""
        return await pinecone.asimilarity_search(query, namespace=self.namespace)
    async def question(self, query:str):
        """Asks a question to the vector store"""
        results = [result.dict(exclude_none=True) for result in await self.search(query)]
        return (await openai_chat.agenerate(messages=[[SystemMessage(content="Similar results:"+str(results))], [HumanMessage(content=query)]])).dict()["generations"][0][0]["text"]