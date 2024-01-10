import chromadb
from langchain.vectorstores import Chroma
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

embedding_function = OpenAIEmbeddings()
db_connection = Chroma(persist_directory='../mk_ultra',embedding_function=embedding_function)

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db_connection.as_retriever())

docs = db_connection.similarity_search('When was this declassified?')
print(docs[0])

compressed_docs = compression_retriever.get_relevant_documents("When was this declassified?")
print(compressed_docs[0].page_content)
