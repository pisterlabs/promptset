#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import os
import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp

import chromadb
from chromadb.config import Settings

from .docs_store import DocsStore


class ChromaDocsStore(DocsStore):
    """

    """

    def __init__(self, config: Dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config, **kwargs)

        #  Load environment variables
        self.target_source_chunks: int = int(config['TARGET_SOURCE_CHUNKS'])
        self.n_ctx = int(config['MODEL_N_CTX'])
        self.n_gpu_layers = int(config['MODEL_N_GPU_LAYERS'])

        # self.splitter = RecursiveCharacterTextSplitter(
        #    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # self.embeddings = HuggingFaceEmbeddings(
        #    model_name=self.embeddings_model_name)

        # self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Define the Chroma settings
        # self.chroma_settings = Settings(
        #    persist_directory=self.persist_directory,
        #    anonymized_telemetry=False
        #)

        self.db = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False))

        # self.db = Chroma(persist_directory=self.persist_directory,
        #                 embedding_function=self.embeddings, client_settings=self.chroma_settings)
        # self.retriever = self.db.as_retriever(
        #    search_kwargs={"k": self.target_source_chunks})

        return

    def vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(self.persist_directory, 'index')):
            if os.path.exists(os.path.join(self.persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(self.persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(
                    os.path.join(self.persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(
                    os.path.join(self.persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False
