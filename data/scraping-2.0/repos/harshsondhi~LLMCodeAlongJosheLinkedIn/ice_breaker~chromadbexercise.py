import sys

import chromadb
import pandas
import sqlite3
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
# openai.api_key = "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"
# api_key = "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"
# llm = OpenAI()
# chat = ChatOpenAI(openai_api_key=api_key)
# print(os.path.dirname(sys.executable))
loader = TextLoader("some_data/FDR_State_of_Union_1944.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# print(docs)
embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(
    docs, embedding_function, persist_directory="./speech_embedding_db"
)
db.persist()

db_new_connection = Chroma(
    persist_directory="./speech_embedding_db", embedding_function=embedding_function
)

new_doc = "what did fdr say about the cost of food law?"

similar_doc = db_new_connection.similarity_search(new_doc)
# print(similar_doc[0])

loader1 = TextLoader("some_data/Lincoln_State_of_Union_1862.txt")
documents1 = loader.load()

text_splitter1 = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs1 = text_splitter1.split_documents(documents1)

db = Chroma.from_documents(
    docs1, embedding_function, persist_directory="./speech_embedding_db"
)

docs1 = db.similarity_search("slavery")
# print(docs1[0].page_content)


retriever = db_new_connection.as_retriever()
search_kwargs = {"score_threshold": 0.8, "k": 4}
docs0 = retriever.get_relevant_documents("President", search_kwargs=search_kwargs)
print(len(docs0))
print(docs0[0].page_content)
