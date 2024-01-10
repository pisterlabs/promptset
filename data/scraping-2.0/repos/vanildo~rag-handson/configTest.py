from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from genai.extensions.langchain import LangChainInterface
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Weaviate
from genai.model import Credentials, Model
from genai.schemas import GenerateParams
from logging.config import dictConfig
from langchain import PromptTemplate
from genai import PromptPattern
from dotenv import load_dotenv
from logging import Logger
import gradio as gr
import unicodedata
import weaviate
import readchar
import chromadb
import logging
import signal
import uuid
import yaml
import os

if __name__ == '__main__':
    print('Sucesso importando dependências')