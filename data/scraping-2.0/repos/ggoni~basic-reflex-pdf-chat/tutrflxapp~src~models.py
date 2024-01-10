from functools import partial
import glob
import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from typing import List
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI


def dumb_answer(question: str) -> str:
    return "¿Me podría repetir la pregunta"


def qa_machine(query: str, vector_store) -> str:
    """
    Perform question answering on the vector store.

    Args:
        query (str): The query for question answering.
        vector_store (FAISS): The vector store.

    Returns:
        str: The answer to the query.
    """

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever())

    result = qa({"query": query})

    return result['result']
