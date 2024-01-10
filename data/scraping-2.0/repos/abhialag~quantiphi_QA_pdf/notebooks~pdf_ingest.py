#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !python --version


# In[1]:


import langchain
import chromadb


# In[ ]:


# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# import click


# In[11]:


import logging
import os

import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader


# In[18]:


# In[12]:


# In[16]:


import constants
from constants import CHUNK_SIZE,CHUNK_OVERLAP, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE,PERSIST_DIRECTORY,EMBEDDING_MODEL_NAME,CHROMA_SETTINGS,ROOT_DIRECTORY


# In[5]:


print(constants.CHUNK_SIZE, PERSIST_DIRECTORY)
print("PERSIST_DIRECTORY",PERSIST_DIRECTORY)
print("ROOT DIRECTORY",ROOT_DIRECTORY)





# In[6]:


def load_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    pages = loader.load()
#     if len(pages)>0:
    print("Loaded {} of pages from the pdf file".format(len(pages)))
    print(pages[0].metadata)
    return pages


# In[7]:


def ret_text_splitters(chunk_size,chunk_overlap,parent_chunk_size,child_chunk_size):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
    # This text splitter is used to create the child documents. It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
    # The vectorstore to use to index the child chunks
    
    return text_splitter, parent_splitter, child_splitter


# In[18]:


def main():
# Load pdf document and split in chunks
    logging.info(f"Loading documents from {constants.SOURCE_DIR_FILE}")
    pages = load_pdf(constants.SOURCE_DIR_FILE)
    text_splitter, parent_splitter, child_splitter = ret_text_splitters(CHUNK_SIZE,
                                                                        CHUNK_OVERLAP, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE)
    texts = text_splitter.split_documents(pages)
    parent_texts = parent_splitter.split_documents(pages)
    child_texts = child_splitter.split_documents(pages)
    logging.info(f"Split into {len(texts)} chunks of text")
    logging.info(f"Split into {len(parent_texts)} chunks of text")
    logging.info(f"Split into {len(child_texts)} chunks of text")
    print(pages[4].page_content)

    # Create embeddings
    # embeddings_instruct = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create in-memory and persist VectorDBs
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

    faiss_vector_db = FAISS.from_documents(texts, embedding=embeddings)
    faiss_vector_db.save_local(f"{PERSIST_DIRECTORY}/faiss.parquet")
        


# In[ ]:


if __name__ == "__main__":
    main()

