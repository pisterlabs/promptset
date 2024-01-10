"""Question-answering with sources over a vector database."""

import warnings
from typing import Any, Dict, List

from pydantic import Field, root_validator

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain.prompts.base import BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT,
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)


class MilvusVectorDBQAWithSourcesChain(BaseQAWithSourcesChain):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore = Field(exclude=True)
    """Vector Database to connect to."""
    k: int = 4
    """Number of results to return from store"""
    reduce_k_below_max_tokens: bool = True
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""


    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        # TODO: replace the LLM of llm_combine_chain to chatgpt3.5?
        llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            combine_document_chain=combine_results_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            **kwargs,
        )

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_documents_chain.llm_chain.llm.get_num_tokens(
                    doc.page_content
                )
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        query = inputs["question"]
        text_ids = inputs["text_ids"]
        store_name = inputs["store_name"]
        docs = self.vectorstore.similarity_search(
            query, text_ids=text_ids, k=self.k, store_name=store_name, **self.search_kwargs
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        raise NotImplementedError("VectorDBQAWithSourcesChain does not support async")

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        warnings.warn(
            "`VectorDBQAWithSourcesChain` is deprecated - "
            "please use `from langchain.chains import RetrievalQAWithSourcesChain`"
        )
        return values

    @property
    def _chain_type(self) -> str:
        return "vector_db_qa_with_sources_chain"
