"""Chain for question-answering against a vector database."""
from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
import json

from pydantic import Extra, Field, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_ChatGLM.stuffchain import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel, BaseRetriever, Document
from langchain.vectorstores.base import VectorStore


class BaseRetrievalQA(Chain):
    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_source_documents: bool = False
    """Return the source documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="Context:\n{page_content}"
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )

        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm, chain_type=chain_type, **_chain_type_kwargs
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @abstractmethod
    def _get_docs(self, question: str) -> List[Document]:
        """Get documents to do question answering over."""

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        question = inputs[self.input_key]

        docs = self._get_docs(question)
        answer, _ = self.combine_documents_chain.combine_docs(docs, question=question)

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    def combine_docs(self, inputs: Dict[str, str], docs: List[Document]) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        question = inputs[self.input_key]
        answer, _ = self.combine_documents_chain.combine_docs(docs, question=question)

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    @abstractmethod
    async def _aget_docs(self, question: str) -> List[Document]:
        """Get documents to do question answering over."""

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        question = inputs[self.input_key]

        docs = await self._aget_docs(question)
        answer, _ = await self.combine_documents_chain.acombine_docs(
            docs, question=question
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class RetrievalQA(BaseRetrievalQA):
    """Chain for question-answering against an index.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain.chains import RetrievalQA
            from langchain.faiss import FAISS
            vectordb = FAISS(...)
            retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=vectordb)

    """

    def _get_docs(self, question: str, url: str) -> List[Document]:

        # url = "http://10.102.33.118:3202/"
        top_k = 2
        print("query:", question)
        data = {"query": question, "top_k": top_k}

        docs = requests.post(url, json=data)
        docs = json.loads(docs.content)
        docs = docs['docs']
        # docs = ["大陆梁子的灾害发生时间为2004年4月6日","大陆梁子隧道的灾害情况是突泥突水"]
        return docs

    async def _aget_docs(self, question: str, url: str) -> List[Document]:
        # url = "http://10.102.33.118:3202/"
        top_k = 2
        print("query:", question)
        data = {"query": question, "top_k": top_k}

        docs = requests.post(url, json=data)
        docs = json.loads(docs.content)
        docs = docs['docs']
        return docs