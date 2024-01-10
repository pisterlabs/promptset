from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import shutil
from git import Repo
from services import DocumentService


class DatabaseHandler:
    persist_directory = 'db'

    embeddings = OpenAIEmbeddings()
    loader = TextLoader('./memory/hi.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)
    else:
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()
