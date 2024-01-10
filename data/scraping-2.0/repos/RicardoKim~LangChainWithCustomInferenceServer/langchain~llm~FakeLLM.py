import requests
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class FakeLLM(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        Self,
        prompt: str,  
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = requests.get("http://localhost:8080/query", json={'query': prompt})
        return response.json()['assistant']
    
    # @property
    # def _identifying_params(self):
    #     """Get the identifying parameters."""
    #     return 