from typing import Any, Dict, List, Optional

from langchain.chains.api.base import APIChain as LangChainAPIChain
from langchain.llms.base import BaseLLM
from langchain.requests import RequestsWrapper

import registry

@registry.register_class
class IndexAPIChain(LangChainAPIChain):
    api_docs_key: str = "api_docs"
    headers_key: str = "headers"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Override the base function to reset instance variables to use relevant input values at query time"""

        api_docs = inputs[self.api_docs_key]
        headers = inputs.get(self.headers_key, None)

        self.api_docs = api_docs
        self.requests_wrapper = RequestsWrapper(headers=headers)
        return super()._call(inputs)
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        **kwargs: Any,
    ) -> LangChainAPIChain:
        return super().from_llm_and_api_docs(
            llm=llm,
            api_docs="",
            **kwargs
        )
