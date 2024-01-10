# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import lru_cache
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from app.constant import LLM_MODEL_3, LLM_MODEL_4, RETRIEVAL_CHAIN
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class HRBot:
    def __init__(self, vector_db: Any, llm_model: ChatOpenAI = LLM_MODEL_3):
        self.vector_db = vector_db
        self.llm_model = llm_model
        self.retriever = self.vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.3})
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain_qa = ConversationalRetrievalChain.from_llm(
            LLM_MODEL_4,
            self.retriever,
            memory=self.memory,
            condense_question_llm=LLM_MODEL_4,
        )

    def setup_bm25(self):
        all_docs = self.vector_db.get()["documents"]
        self.bm25_retriever = BM25Retriever.from_texts(all_docs)
        self.bm25_retriever.k = 2

    def ensemble_search_docs(self):
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retriever], weights=[0.5, 0.5])

    def search_docs(self, query_input, embedding_func):
        return self.vector_db.similarity_search_by_vector(embedding_func.embed_query(query_input), k=1)

    @lru_cache
    def generate_response(self, query: str):
        # relevant_docs = self.ensemble_retriever.get_relevant_documents(query)
        # print(relevant_docs)
        return self.chain_qa({"question": query})
        # return self.chain_qa.run({
        #    "question": query,
        #    "context": relevant_docs
        # })
