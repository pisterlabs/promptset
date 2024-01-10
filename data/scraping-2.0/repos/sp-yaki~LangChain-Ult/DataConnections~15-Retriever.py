import chromadb
from langchain.vectorstores import Chroma
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

embedding_function = OpenAIEmbeddings()
db_connection = Chroma(persist_directory='./mk_ultra',embedding_function=embedding_function)
retriever = db_connection.as_retriever()

search_kwargs = {"score_threshold":0.8,"k":4}
docs = retriever.get_relevant_documents("President",
                                       search_kwargs=search_kwargs)

print(len(docs))


loader = WikipediaLoader(query='MKUltra')
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function,persist_directory='./mk_ultra')
db.persist()

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
question="When was this declassified?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(),llm=llm)

# Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)

print(unique_docs[0].page_content)