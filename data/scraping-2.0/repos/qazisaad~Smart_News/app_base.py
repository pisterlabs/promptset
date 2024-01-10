import asyncio
import aiohttp

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
from time import sleep
import streamlit as st
from abc import ABC, abstractmethod
from typing import List, Tuple
from time import time

from Classes.classes import DuckDuckGoNews, Page

import requests

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

async def process_results(results):
  """
   Parses the results in parallel using asyncio and returns a list of tuples containing
    the heading, description, and title for each page.
    :param results: a tuple containing lists of result links, headings, and descriptions
    :return: a list of tuples containing the heading, description, and title for each page
  """
  pages = []
  tasks = []
  async with aiohttp.ClientSession() as session:
    for link, heading, desc in zip(*results):
      page = Page(link)
      task = asyncio.ensure_future(page.parse(True))
      tasks.append(task)
    pages = await asyncio.gather(*tasks)
  return pages


def setup_crawler():
  crawler = DuckDuckGoNews()
  return crawler

t = time()
crawler = setup_crawler()
print('Time for setting crawler', time() - t)


t = time()
results = crawler.get_results(query)
print('Time for scraping search page', time() - t)

t = time()
results = asyncio.run(process_results(results))
# results = await results
results = list(results)
print('results', results)
print('Time for scraping search pages results', time() - t)


t = time()
os.environ["OPENAI_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

texts = [x.text for x in  results]

ids = [{'source': i} for i, string in enumerate(texts)]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20, length_function = tiktoken_len, separators=['\n\n', "\n", ' ', ''])
texts = text_splitter.create_documents(texts, metadatas=ids)
print('Time for splitting text', time() - t)

t = time()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
print('Time for setting up database', time() - t)


t = time()
chat = ChatOpenAI(temperature=0)
# llm_flan_t5 =HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
# llm_bart =HuggingFaceHub(repo_id="vblagoje/bart_lfqa")
# llm = OpenAI()
retriever = docsearch.as_retriever()
qa_source = RetrievalQAWithSourcesChain.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
# qa = RetrievalQA.from_chain_type(llm=llm_flan_t5, chain_type="stuff", retriever=retriever)
print('Time for setting up QA model', time() - t)

t = time()
print(qa_source({'question':query}, return_only_outputs=True))
print('Time for predictions', time() - t)

