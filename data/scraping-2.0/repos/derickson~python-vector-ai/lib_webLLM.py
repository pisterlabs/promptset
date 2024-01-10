from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


import requests


class WebLLM(LLM):
    
    ## defaults to local.  you can override this in constructor
    url: str = "http://nuc:8090/question"

    @property
    def _llm_type(self) -> str:
        return "WebLLM"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        payload = {
            "question": prompt
        }
        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            # print(response.json())
            return str(response.json()["answer"][0])
        else:
            return "error occurred"
        
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url}