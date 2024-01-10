from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import os
from data_space import *


def store_pdf(file_name, user_id):
    try:
        file_path = os.path.join(sys.path[0], 'files', user_id, file_name)
        loader = PyPDFLoader(file_path)
        store_document(loader, user_id, file_name)
        return True
    except Exception as e:
        print(e)
        return False


def store_word(file_name, user_id):
    try:
        file_path = os.path.join(sys.path[0], 'files', user_id, file_name)
        loader = Docx2txtLoader(file_path)
        store_document(loader, user_id, file_name)
        return True
    except Exception as e:
        print(e)
        return False


def store_text(file_name, user_id):
    try:
        file_path = os.path.join(sys.path[0], 'files', user_id, file_name)
        loader = TextLoader(file_path, encoding='utf-8')
        store_document(loader, user_id, file_name)
        return True
    except Exception as e:
        print(e)
        return False


def store_pptx(file_name, user_id):
    try:
        file_path = os.path.join(sys.path[0], 'files', user_id, file_name)
        loader = UnstructuredPowerPointLoader(file_path)
        store_document(loader, user_id, file_name)
        return True
    except Exception as e:
        print(e)
        return False


def store_document(loader, user_id, file_name):
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    ids = [file_name+str(i) for i in range(1, len(docs) + 1)]
    add2json(user_id, file_name, 1, len(ids)+1)

    embeddings = FakeEmbeddings(size=1352)
    persist_directory = os.path.join(sys.path[0], 'db')
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    memorydb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory, collection_name=user_id, ids=ids)
    memorydb.persist()
    memorydb = None
