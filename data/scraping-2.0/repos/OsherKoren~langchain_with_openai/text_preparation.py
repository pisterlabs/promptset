# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module contains the prompts for the use cases."""


from typing import List, Union

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
import connect, utils

log = utils.setup_logger()


def get_openai_embedding(text: str, model="text-embedding-ada-002", text_type: str = "query"):
    embedding = OpenAIEmbeddings(model=model)
    if text_type == "query":
        embedded_text = embedding.embed_query(text)
    elif text_type == "doc":
        embedded_text = embedding.embed_doc(text)
    else:
        raise NotImplementedError(f"Text type {text_type} not implemented.")

    log.info(f"Embedding length is: {len(embedded_text)}")
    return embedded_text


def get_openai_embedding_list(docs: List[Document], model="text-embedding-ada-002"):
    embedding = OpenAIEmbeddings(model=model)
    embedding_list = embedding.embed_documents([doc.page_content for doc in docs])

    log.info(f"Embedding list length is: {len(embedding_list)}")
    return embedding_list


def split_docs_recursively(docs: Union[str, List[Document]], chunk_size: int = 1000, chunk_overlap: int = 20):
    """Split text into chunks of characters."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    if isinstance(docs, str):
        docs = text_splitter.create_documents([docs])
    elif isinstance(docs, list):
        docs = text_splitter.split_documents(docs)
    else:
        raise NotImplementedError(f"Type {type(docs)} not implemented.")

    log.info(f"Number of documents: {len(docs)}")
    return docs


def retrieve_docs(docs: List[Document],
                  model="text-embedding-ada-002",
                  chunk_size: int = 1000,
                  chunk_overlap: int = 20):
    """Retrieve and embed documents."""
    texts = split_docs_recursively(docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding = OpenAIEmbeddings(model=model)
    db = FAISS.from_documents(texts, embedding)
    retriever = db.as_retriever()
    return retriever


def extract_relevant_docs(
        text: str,
        docs: List[Document],
        model="text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 20):
    """Retrieve and get relevant documents."""
    retriever = retrieve_docs(docs=docs, model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    relevant_docs = retriever.get_relevant_documents(text)
    log.info(f"Number of relevant documents: {len(relevant_docs)}")
    return relevant_docs
