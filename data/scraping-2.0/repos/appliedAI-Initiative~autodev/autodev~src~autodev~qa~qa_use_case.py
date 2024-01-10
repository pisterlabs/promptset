"""
Simple question answering use case implementation based on a (fixed) document database
"""
import logging
from typing import List

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter

from autodev.llm import LLMType
from .document_db import DocumentDatabase, VectorDatabase

log = logging.getLogger(__name__)


class QuestionAnsweringUseCase:
    """
    Represents a question answering use case
    """
    def __init__(self, llm_type: LLMType, doc_db: DocumentDatabase, splitter: TextSplitter, queries: List[str]):
        """
        :param llm_type: An LLMType object, which specifies the type of LLM model to use.
        :param doc_db: A DocumentDatabase object, which contains the text documents for querying.
        :param splitter: A TextSplitter object, which is used to split the documents into sub-documents.
        :param queries: A list of strings, representing example queries that can be executed.
        """
        self.llm_type = llm_type
        self.doc_db = doc_db
        self.queries = queries
        self.splitter = splitter
        self.vector_db = VectorDatabase(doc_db.name, doc_db, splitter, OpenAIEmbeddings())
        log.info(f"Creating model {llm_type}")
        llm = llm_type.create_llm()
        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vector_db.retriever())

    def query(self, q):
        print(f"\n{q}")
        answer = self.qa.run(q)
        print(answer)

    def run_example_queries(self):
        for q in self.queries:
            self.query(q)
