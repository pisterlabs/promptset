# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description:

import streamlit as st
from streamlit_extras.stateful_chat import chat, add_message
from streamlit_extras.streaming_write import write as streamlit_write


from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler


from dotenv import load_dotenv
import os
import time

import sys
import pathlib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="ebooklib.*")


def main():
    load_dotenv()
    data_path = os.getenv("BOOK_PATH")
    raw_documents = {}
    for item in epub.read_epub(data_path, {"ignore_ncx": False}).get_items():
        chapter = "Unknown"
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(
                item.get_body_content().decode('utf-8'), "html.parser")     
            if soup.find("h1"):
                chapter = soup.find("h1").get_text()     
            elif soup.find("h2"):
                chapter = soup.find("h2").get_text()
            paragraphs = soup.find_all("p")
            for paragraph in paragraphs:
                if raw_documents.get(chapter):
                    texts = raw_documents.get(chapter)
                    texts.append(paragraph.get_text())
                    raw_documents.update({
                        chapter: texts
                    })
                else:
                    raw_documents.update({
                        chapter: [paragraph.get_text()]
                    })
    for doc in raw_documents:
        print(doc, len(raw_documents[doc]))

if __name__ == '__main__':
    main()
