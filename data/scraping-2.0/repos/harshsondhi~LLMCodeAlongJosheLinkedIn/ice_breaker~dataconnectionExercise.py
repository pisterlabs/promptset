import sys

import chromadb
import pandas
import sqlite3
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WikipediaLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
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
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
# chat = ChatOpenAI(openai_api_key=api_key, temperature=0)
embedding_function = OpenAIEmbeddings()


def us_constitution_helper(question):
    loader = TextLoader("some_data/US_Constitution.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)

    _embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(
        docs, _embedding_function, persist_directory="./US_Constitution"
    )
    db.persist()

    chat = ChatOpenAI(openai_api_key=api_key, temperature=0)
    compressor = LLMChainExtractor.from_llm(chat)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=db.as_retriever()
    )

    compressed_docs = compression_retriever.get_relevant_documents(question)

    return compressed_docs[0].page_content


print(us_constitution_helper("What is the 13th Amendment?"))
