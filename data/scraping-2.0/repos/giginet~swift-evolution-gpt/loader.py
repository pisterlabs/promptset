from functools import reduce
from pathlib import Path
from typing import List

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext, StorageContext, \
    load_index_from_storage, LLMPredictor, OpenAIEmbedding, download_loader, Document
from llama_index.indices.base import IndexType
from llama_index.llms import OpenAI

import chromadb
from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore

import os
import logging

from llama_index.readers.file.markdown_reader import MarkdownReader


class ProposalsLoader:
    @property
    def cache_path(self) -> str:
        return os.path.join(os.getcwd(), ".caches")

    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.llm = OpenAI(model='gpt-4-1106-preview')

    def load(self) -> IndexType:
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        predictor = LLMPredictor(llm=self.llm)
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            llm_predictor=predictor
        )
        # documents = SimpleDirectoryReader(self.directory_path).load_data()

        markdown_reader = MarkdownReader()
        proposals = [os.path.join(self.directory_path, markdown)
                     for markdown in os.listdir(self.directory_path) if markdown.endswith(".md")]

        def extend_markdowns(list: List[Document], filepath: str) -> List[Document]:
            docs = markdown_reader.load_data(file=Path(filepath))
            list.extend(docs)
            return list

        documents: List[Document] = reduce(
            extend_markdowns,
            proposals,
            []
        )

        db = chromadb.PersistentClient(path="./chroma_db")

        chroma_collection = db.get_or_create_collection("swift-evolution-gpt")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = GPTVectorStoreIndex.from_documents(documents,
                                                   service_context=service_context,
                                                   storage_context=storage_context)
        return index