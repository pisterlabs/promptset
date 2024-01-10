
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from .prompt import PROMPT
import os

from pathlib import Path

# get the absolute path of the current script
script_location = Path(os.path.abspath(__file__))

# get the directory containing the script
rootdir = script_location.parents[2]

# construct the full path to the config file
CHROMA_DB_DIR = str(rootdir / 'data' / 'db')
DOCUMENT_DIR = str(rootdir / 'data' / 'site_contents')

def get_collection():
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(
        collection_name=os.environ['CHROMADB_COLLECTION_NAME'],
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return docsearch

def load():
    docsearch = get_collection()
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model_name=os.environ['OPENAI_MODEL']),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )
    chain.return_source_documents = True
    return chain

def create_chroma_db():
    collection = get_collection()
    collection.delete_collection()
    loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, silent_errors=True)
    docs = []
    docs.extend(loader.load())
    sub_docs = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500).split_documents(docs)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(
        sub_docs,
        embeddings,
        collection_name=os.environ['CHROMADB_COLLECTION_NAME'],
        persist_directory=CHROMA_DB_DIR
    )
    docsearch.persist()
    return docsearch

    