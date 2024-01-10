import os
import getpass
import tempfile

#import streamlit as st

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings)

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# os.environ['OPENAI_API_KEY'] = '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' #getpass.getpass('OpenAI API Key:')

"""
https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
"""
#----------------------------------------------------------------------------------------------------------#
# Document Loaders 
# uploaded_files = './layout-parser-paper.pdf'

# loader = PyPDFLoader(uploaded_files)
# pages = loader.load_and_split() # 해당 방식은 문서를 페이지 번호로 검색할 수 있다.
# pages_d = loader.load()
# print(pages[0])
# """
# Document(page_content='', type='Document', metadata={'source': './layout-parser-paper.pdf', 'page': 0}) 의
# 리스트가 담겨있다.
# """

# loader2 = PyPDFLoader(uploaded_files, extract_images=True)
# pages2 = loader2.load()
# print(pages2[3].page_content)

#----------------------------------------------------------------------------------------------------------#
# Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 100,
#     chunk_overlap  = 20,
#     length_function = len,
#     add_start_index = True,
# )

# chunk_size = 1000
# chunk_overlap = 150
# r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# docs = r_splitter.split_documents(pages)
# print(docs)

# texts = text_splitter.create_documents([state_of_the_union])
# print(texts[0])
# print(texts[1])
#----------------------------------------------------------------------------------------------------------#
# faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
#----------------------------------------------------------------------------------------------------------#
import os, time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv

from pymilvus import connections
from pymilvus import Collection, CollectionSchema
from pymilvus import FieldSchema, DataType
from pymilvus import utility


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import sys
sys.path.append('D:/203_GenAI_IBM/Manual')

# ## text splitter parmas
# chunk_size = 1000
# chunk_overlap = 150
# separator = "\n"

# ## create text splicter instance
# r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# filename = './[국문 G90 2023] genesis-g90-manual-kor-230601.pdf'
# loader = PyPDFLoader(filename)
# pages = loader.load()
# print('len(pages): ', len(pages))

# index = 23

# docs = r_splitter.split_documents([pages[index]])
# print(docs)

from sentence_transformers import SentenceTransformer
sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
print(model)
# embeddings = model.encode(sentences)
# print(embeddings)

from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/all-MiniLM-L6-v2'          #embedding 모델 변경 test
embeddings2 = HuggingFaceEmbeddings(model_name=model_name)
print(embeddings2)
