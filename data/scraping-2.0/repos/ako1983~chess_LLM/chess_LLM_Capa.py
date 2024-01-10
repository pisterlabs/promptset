#!/usr/bin/env python
# coding: utf-8

# # Chess Knowledge Extractor
# 
# This project is designed to extract and analyze chess-related knowledge from various sources. It aims to answer questions like "What is the best way to learn chess?" based on advice from the legendary José Raúl Capablanca and from popular chess resources like 'Say Chess' Substack.

import dotenv
import os
import re
from bs4 import BeautifulSoup
import requests
import json
from collections import defaultdict

import openai
from langchain.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv, find_dotenv

# Load env vars from .env file
load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to download text file from a URL
def download_text_file(url):
    response = requests.get(url)
    return response.text

# Function to read text file
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


# Function to preprocess text
import re

def preprocess_text(text):
    """
    This function takes in a string of text and removes any extra whitespaces.

    Args:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text with extra whitespaces removed.
    """
    text = re.sub(r'\s+', ' ', text)
    return text
# # Part1: Chess Fundamentals, by Capablanca

# URL of the Gutenberg project book
book_url = 'https://www.gutenberg.org/files/33870/33870-8.txt'  # EBook of Chess Fundamentals, by Capablanca

# Download and read the text file
text_from_web = download_text_file(book_url)

# Optionally, you can save this to a local text file
with open("downloaded_book.txt", "w") as f:
    f.write(text_from_web)

# Preprocess the text
preprocessed_text = preprocess_text(text_from_web)


llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

qa_document_chain.run(input_document=preprocessed_text, question="what is the best way to learn chess")

# # Part 2: Say Chess Substack.

# read from web and answer question about chess
## 10 simple pieces of chess advice

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://saychess.substack.com/p/10-simple-pieces-of-chess-advice")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(openai_api_key=openai.api_key))
retriever = vectorstore.as_retriever()


from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

# LLM

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1,  )

# RAG chain 

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rag_chain.invoke("10-simple-pieces-of-chess-advice")