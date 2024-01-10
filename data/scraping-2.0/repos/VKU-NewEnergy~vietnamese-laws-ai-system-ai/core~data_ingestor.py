"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import time


from typing import List, Text, Tuple

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS

from core.constants import IngestDataConstants
from llm.base_model.langchain_openai import openai_embedding_with_backoff

class DataIngestor:
    """Ingest data with different format to create vectorstore"""
    def __init__(self, lang: str = ""):
        self.lang = lang
        self.vectorstore_path = IngestDataConstants.VECTORSTORE_FOLDER

    def create_vectorstore(self) -> Tuple[Text, Text]:
        try:
            if not os.path.exists(self.vectorstore_path):
                os.makedirs(self.vectorstore_path)

        except Exception as e:
            logging.exception(e)
        finally:
            return self.vectorstore_path, self.sensor_data_lib_path
        
    def _save_vectorstore(self, raw_documents: List, vectorstore_path: Text, id: str = None):
        text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=IngestDataConstants.CHUNK_SIZE,
            chunk_overlap=IngestDataConstants.CHUNK_OVERLAP,
        )

        splitted_documents = text_splitter.split_documents(raw_documents)
        embeddings = openai_embedding_with_backoff()

        if os.path.exists(f"{vectorstore_path}/index.faiss"):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings)
            if id:
                vectorstore.add_documents(splitted_documents, ids=[id])
            else:
                vectorstore.add_documents(splitted_documents)
        else:
            texts = [doc.page_content for doc in splitted_documents]
            if id:
                vectorstore = FAISS.from_texts(texts, embeddings, ids=[id])
            else:
                vectorstore = FAISS.from_texts(texts, embeddings)
        
        vectorstore.save_local(vectorstore_path)
        time.sleep(60)

    def load_charter(self, plain_text: Text, id: str):
        vectorstore_path = self.create_vectorstore()[1]
        txt_path = os.path.join(vectorstore_path, f"{id}", "content.txt")

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf8") as fi:
            fi.write(plain_text)

        self.ingest_charter(txt_path, id)

    def ingest_charter(self, txt_path: Text, id: str):
        vectorstore_path = self.create_vectorstore()[1]

        loader = UnstructuredFileLoader(txt_path)
        raw_documents = loader.load()

        self._save_vectorstore(raw_documents, vectorstore_path, id)
        time.sleep(10)
