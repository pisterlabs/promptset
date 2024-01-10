import os
import logging
import pathlib

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader


class AssetHelper:
    _path_asset = str(pathlib.Path(__file__).parent.parent.resolve() / 'data')
    _path_db = str(pathlib.Path(__file__).parent.parent.resolve() / 'chroma-persist')
    _collection_name = 'kakao-api'
    _file_extensions = ['txt']
    _loader_dict = {
        'txt': TextLoader,
    }

    _chunk_size = 500
    _chunk_overlap = 100

    def __init__(self, path_asset=None, file_extensions=None, path_db=None, collection_name=None):
        if path_asset:
            self._path_asset = str(path_asset)
        if file_extensions:
            self._file_extensions = file_extensions
        if path_db:
            self._path_db = str(path_db)
        if collection_name:
            self._collection_name = collection_name
        self._openai_embeddings = OpenAIEmbeddings()

        self._db = Chroma(
            persist_directory=self._path_db,
            embedding_function=self._openai_embeddings,
            collection_name=self._collection_name,
        )
        self._retriever = self._db.as_retriever()

    def load(self, force_reload=False):
        if len(self._db.get()['documents']) > 0:
            if not force_reload:
                logging.info('db already exists')
                return
            else:
                logging.info('remove db')
                os.remove(self._path_db)
        for root, dirs, files in os.walk(self._path_asset):
            for file in files:
                ext = file.split('.')[-1]
                if ext in self._file_extensions:
                    path_file = os.path.join(root, file)
                    loader = self._loader_dict[ext]
                    self._load_file(path_file, loader)

    def _load_file(self, path_file, loader):
        logging.info('load file: %s', path_file)
        documents = loader(path_file).load()
        text_splitter = CharacterTextSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        docs = text_splitter.split_documents(documents)
        logging.info(docs)

        Chroma.from_documents(
            docs,
            self._openai_embeddings,
            collection_name=self._collection_name,
            persist_directory=self._path_db,
        )

    def query(self, input):
        return self._retriever.get_relevant_documents(input)
