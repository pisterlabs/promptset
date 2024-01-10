# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Base module for natural language retrieval."""
from abc import ABC, abstractmethod
import logging
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore
from langchain.memory import ConversationSummaryBufferMemory

from vector_stores import read_vector_store
from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make
up an answer.

{context}

Question: {question}

"""


class RetrievalBase(ABC):
    """Base class for langchain retrieval classes."""

    _PROMPT_TEMPLATE = PROMPT_TEMPLATE

    def __init__(
        self,
        vs_path: str,
        max_tokens: int = 2000,
        memory: bool = False,
        verbose: bool = False,
        model_name: str = "gpt-3.5-turbo",
        **kwargs,
    ):
        """Initialize the retriever."""
        try:
            vectorstore = read_vector_store(
                vs_path=vs_path, caller=f"{self.__class__.__name__}.create_vectorstore"
            )
        except ValueError:
            logging.warning("Vector store not found - creating new one")
            vectorstore = self.create_vectorstore(vs_path=vs_path, **kwargs)

        if memory:
            self.chat_memory = ConversationSummaryBufferMemory(llm=OpenAI())
        else:
            self.chat_memory = None

        self.retriever = self.create_qa_retriever(
            model_name=model_name,
            vectorstore=vectorstore,
            max_tokens=max_tokens,
            memory=self.chat_memory,
            verbose=verbose,
        )

    @property
    def current_memory(self):
        """Display the memory."""
        if self.chat_memory:
            self.chat_memory.load_memory_variables({})

    def ask(self, question: str):
        """Ask a question."""
        return self.retriever(question)["result"]

    def ask_conversation(self, question: str):
        """Ask a question getting the full response."""
        return self.retriever(question)

    def create_qa_retriever(
        self,
        vectorstore: VectorStore,
        max_tokens: int = 2000,
        verbose: bool = False,
        memory: Optional[ConversationSummaryBufferMemory] = None,
        model_name: str = "gpt-3.5-turbo",
    ):
        """Create a retriever."""
        prompt = PromptTemplate(
            template=self._PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}

        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=model_name, temperature=0, max_tokens=max_tokens),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            memory=memory,
            verbose=verbose,
        )

    @classmethod
    @abstractmethod
    def create_vectorstore(cls, input_path, vs_path: str):
        """Create a vector store."""
