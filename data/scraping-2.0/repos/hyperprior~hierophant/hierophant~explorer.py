from hierophant import statistics
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import json
import logging
import os
from datetime import datetime
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import ray
from beartype import beartype
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.document_loaders.base import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.retrievers import SVMRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel
from sklearn import datasets
from ydata_profiling import ProfileReport

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class Explorer:
    def __init__(
        self,
        dataset,
        dataset_operations: Optional[list] = None,
        column_operations: Optional[list] = None,
        llm=ChatOpenAI(openai_api_key=os.getenv(
            "OPENAI_API_KEY"), temperature=0),
        vectorstore=FAISS,
        embeddings=OpenAIEmbeddings(),
        return_source_documents: bool = True,
    ):
        self.analysis_time_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        ray.init(ignore_reinit_error=True)
        self.dataset = dataset
        if not dataset_operations:
            self.dataset_operations = [
                self.number_rows,
            ]
        if not column_operations:
            self.column_operations = [
                statistics.column_min,
                (statistics.column_mean, {"sigdig": 4}),
                statistics.column_variance,
                statistics.column_std,
                statistics.column_quantiles,
                statistics.column_max,
                statistics.column_dtype,
                statistics.column_number_negative,
                statistics.column_proportion_negative,
                statistics.column_number_zeros,
                statistics.column_proportion_zeros,
                statistics.column_number_positive,
                statistics.column_proportion_positive,
            ]
        self.llm = llm
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.return_source_documents = return_source_documents
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self._results = []
        for operation in self.dataset_operations:
            self._results.append(operation.remote(
        for column in self.dataset.columns:
            if self.dataset[column].dtype == "object":
                continue
            try:
                for operation in self.column_operations:
                    # self._results.append(
                    #     compute_statistics.remote(column, cancer[column], operation)
                    # )

                    if isinstance(operation, tuple):
                        self._results.append(
                            operation[0].remote(
                                column={"name": column,
                                    "values": self.dataset[column]},
                                **operation[1]
                            )
                        )
                    else:
                        self._results.append(
                            operation.remote(
                                column={"name": column,
                                    "values": self.dataset[column]}
                            )
                        )
            except:
                pass
        self.profile=ray.get(self._results)
        self.profile_documents=[
            Document(page_content=json.dumps(result)) for result in self.profile
        ]
        self.retriever=self.vectorstore.from_documents(
            documents=self.profile_documents, embedding=self.embeddings
        ).as_retriever()
        self.chat=ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.retriever, memory=self.memory
        )

    def explore(self, question):
        return self.chat({"question": question})
