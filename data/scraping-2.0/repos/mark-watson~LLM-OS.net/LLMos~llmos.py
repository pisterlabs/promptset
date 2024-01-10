# Wrapper for OpenAI APIs for llmlib

import openai
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class LLMos():
    def __init__(self, key=None, embeddings_dir="./db_embeddings"):
        self.embeddings_dir=embeddings_dir
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                raise Exception("OPENAI_API_KEY environment variable not set")
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.embeddings_dir = embeddings_dir
        self.db = None
        self.index = None
        self.retriever = None
        self.qa = None

