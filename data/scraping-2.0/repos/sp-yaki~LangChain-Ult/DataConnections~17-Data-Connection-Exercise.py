import chromadb
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor 

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

def us_constitution_helper(question):
    '''
    Takes in a question about the US Constitution and returns the most relevant
    part of the constitution. Notice it may not directly answer the actual question!
    
    Follow the steps below to fill out this function:
    '''
    # PART ONE:
    # LOAD "some_data/US_Constitution in a Document object
    loader = TextLoader("../some_data/US_Constitution.txt")
    documents = loader.load()
    
    # PART TWO
    # Split the document into chunks (you choose how and what size)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)
    
    # PART THREE
    # EMBED THE Documents (now in chunks) to a persisted ChromaDB
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embedding_function,persist_directory='../US_Constitution')
    db.persist()

    # PART FOUR
    # Use ChatOpenAI and ContextualCompressionRetriever to return the most
    # relevant part of the documents.

    # results = db.similarity_search("What is the 13th Amendment?")
    # print(results[0].page_content) # NEED TO COMPRESS THESE RESULTS!
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                           base_retriever=db.as_retriever())

    compressed_docs = compression_retriever.get_relevant_documents(question)

    return compressed_docs[0].page_content

print(us_constitution_helper("What is the 13th Amendment?"))