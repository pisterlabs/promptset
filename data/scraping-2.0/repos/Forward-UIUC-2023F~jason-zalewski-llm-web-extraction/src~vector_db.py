import os
from typing import List
import openai
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import pprint

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader

from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

import web_extractor



# Specify the path to config.json (adjust the path as needed)
config_file_path = '/Users/jasonz/forward_data_lab_llmie/forward_data-llm_ie/config.json'

with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
    openai.api_key = config["api_key"]

llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)



def retrieve_information_from_url(url: str, query: str) -> str:
    
    # Load the document, split it into chunks, embed each chunk, and load it into the vector store
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai.api_key))
    
    # Search for similarity based on the query
    docs_sim = db.similarity_search(query)
    
    # Retrieve and return the page content
    if docs_sim:
        return docs_sim[0].page_content
    else:
        return "Information not found."

# Helper function for printing docs

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def vanila_vector_strore_retriver(url: str, query: str) -> str:
    loader = WebBaseLoader(url)
    documents = loader.load() 
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai.api_key)).as_retriever()

    docs = retriever.get_relevant_documents(query)
    pretty_print_docs(docs)
    return docs

def llm_chain_extractor_retriver(url: str, query: str) -> str:
    loader = WebBaseLoader(url)
    documents = loader.load() 
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai.api_key)).as_retriever()
    
    #llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    compresed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compresed_docs)
    return compresed_docs

def string_compressor_and_document_transformer_retriver(url: str, query: str) -> str:
    loader = WebBaseLoader(url)
    documents = loader.load() 
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai.api_key)).as_retriever()
 

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
    transformers=[text_splitter, redundant_filter, relevant_filter])
    
    #llm = OpenAI(temperature=0)
    #compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    compresed_docs = compression_retriever.get_relevant_documents(query)
    pretty_print_docs(compresed_docs)
    return compresed_docs

# Example usage:
url = "https://cs.illinois.edu/about/people/faculty/jeffe"
query = "What is the research interests of the professor?"

#answer = retrieve_information_from_url(url, query)
#print(answer)

#answer = vanila_vector_strore_retriver(url, query)
##answer = llm_chain_extractor_retriver(url, query)
answer = string_compressor_and_document_transformer_retriver(url, query)
#pretty_print_docs(answer)