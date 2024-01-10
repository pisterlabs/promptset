from typing import List, Optional, Any, Dict

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.pydantic_v1 import Extra
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.language_model import BaseLanguageModel
from langchain.vectorstores.chroma import Chroma


class CVPartnerQA(Chain):
    combine_documents_chain: BaseCombineDocumentsChain

    input_key_list: List[str] = ["query", "email"]  #: :meta private:
    output_key: str = "result"  #: :meta private:

    vector_store: Chroma

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        return self.input_key_list

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        vector_store: Chroma,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Load chain from chain type."""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm, chain_type=chain_type, **_chain_type_kwargs
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            vector_store=vector_store,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["query"]
        email = inputs["email"]

        docs = await self.vector_store.amax_marginal_relevance_search(
            question, k=5, filter={"email": email.lower()}
        )

        answer = await self.combine_documents_chain.arun(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )
        return {self.output_key: answer}
