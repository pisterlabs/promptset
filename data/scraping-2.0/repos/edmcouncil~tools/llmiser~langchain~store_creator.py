from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from llmiser.langchain.config import EMBEDDING_MODEL

MAXIMAL_SPLIT_LENGTH = 4084

def create_and_persist_store(input_docs_folder_path: str, store_path: str):
    loader = GenericLoader.from_filesystem(input_docs_folder_path)
    contents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    all_splits = text_splitter.split_documents(contents)
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=EMBEDDING_MODEL)
    vectorstore.save_local(folder_path=store_path)
