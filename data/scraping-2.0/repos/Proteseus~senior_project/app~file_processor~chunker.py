import os
import pickle
import logging
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredURLLoader

class Chunker:
    def __init__(self, urls=None, file_path=None):
        self.urls = urls or []
        self.file_path = file_path or []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.logger = logging.getLogger(__name__)

    def load_url(self):
        loader_url = UnstructuredURLLoader(urls=self.urls)
        url_load = loader_url.load()
        self.logger.info('URL loaded')
        return url_load

    def load_file(self):
        loader_text = TextLoader(file_path=self.file_path)
        text_load = loader_text.load()
        self.logger.info('Text Loaded')
        return text_load
    
    def chunker(self, loader_type):
        if loader_type == 'text':
            loader = self.load_file()
        elif loader_type == 'url':
            loader = self.load_url()
        else:
            raise ValueError("Invalid loader type")

        split_docs = self.text_splitter.split_documents(loader)
        self.logger.info('Documents Split')
        return split_docs
