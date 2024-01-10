__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from typing import AnyStr, List, TypeVar

from langchain.document_loaders import CSVLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

InstanceType = TypeVar('InstanceType', bound='LLMSearch')

"""
    Wrapper for LLM search of vector stores using FAISS
"""


class LLMEmbeddingSearch(object):
    text_document_type = 'text'
    cvs_document_type = 'csv'
    json_document_type = 'json'
    document_types = [text_document_type, cvs_document_type, json_document_type]

    default_search = 'default_search'
    search_by_vector = 'search_by_vector'

    def __init__(self, documents: List[Document]):
        """
        Constructor for search for embedding vectors
        @param documents: List of documents
        """
        assert documents, 'List of documents cannot be empty'
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(documents, self.embeddings)

    @classmethod
    def build(cls, filenames: List[AnyStr], document_type: AnyStr) -> InstanceType:
        """
        Static constructor, builder using a file name and document type
        @param filenames: Name of the file containing the document
        @param document_type: Type of documents text, csv, json,...
        @return: Instance of this class if document type is supported, a NotImplementedError otherwise
        """
        import itertools

        if document_type == LLMEmbeddingSearch.text_document_type:
            raw_docs = [TextLoader(filename).load() for filename in filenames]
        elif document_type == LLMEmbeddingSearch.cvs_document_type:
            raw_docs = [CSVLoader(filename).load() for filename in filenames]
        elif document_type == LLMEmbeddingSearch.json_document_type:
            import json
            from pathlib import Path
            raw_docs = [json.loads(Path(filename).read_text()) for filename in filenames]
        else:
            raise NotImplementedError(f'Document type {document_type} is not supported')

        raw_documents: List[Document] = raw_docs[0] if len(raw_docs) == 1 \
            else list(itertools.chain.from_iterable(raw_docs))

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents: List[Document] = text_splitter.split_documents(raw_documents)
        return cls(documents)

    def __call__(self, query: AnyStr, search_type: AnyStr) -> List[AnyStr]:
        """
        Execute search using direct, default search of
        @param query: Prompt defined as an instruction
        @param search_type: Type of search: Default or Embedding vector
        @return: List of answers if search_type is supported, Not implemented error otherwise
        """

        if search_type == LLMEmbeddingSearch.default_search:
            return self.__default_search(query)
        elif search_type == LLMEmbeddingSearch.search_by_vector:
            return self.__search_by_vector(query)
        else:
            raise NotImplementedError(f'Search type {search_type} not supported, use default search')

            # ------------------  Supporting methods ---------------------
    def __default_search(self, query: AnyStr) -> List[AnyStr]:
        docs = self.db.similarity_search(query)
        return [doc.page_content for doc in docs]

    def __search_by_vector(self, query: AnyStr) -> List[AnyStr]:
        embedding_vector = self.embeddings.embed_query(query)
        docs: List[Document] = self.db.similarity_search_by_vector(embedding_vector)
        return [doc.page_content for doc in docs]


