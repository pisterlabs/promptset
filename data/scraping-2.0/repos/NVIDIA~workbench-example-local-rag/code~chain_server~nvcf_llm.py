from typing import Any, List, Mapping, Optional
import os

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field
import requests

class NvcfLLM(LLM):
    endpoint: str = Field(None, alias="endpoint")
    max_new_tokens: int = Field(None, alias="max_new_tokens")

    @property
    def _llm_type(self) -> str:
        return "nvcf"

    def _wait_for_fullfill(self, reqID, headers):
        url = "https://api.nvcf.nvidia.com/v2/nvcf/exec/status/" + reqID
        response = requests.get(url, headers=headers)

        content = None
        if response.status_code == 200:
            response_json = response.json()
            status = response_json.get("status")
            if status == "fulfilled":
                content = response_json.get('response', {}).get('choices', [])[0].get('message', {}).get('content', None)
        
        return content

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        token = os.environ.get("NVCF_RUN_KEY")
        headers = {
            "Authorization": f"Bearer {token}",  
            "Content-Type": "application/json"
        }

        data = {
            "requestBody": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        }

        response = requests.post(self.endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            content = response_json.get('response', {}).get('choices', [])[0].get('message', {}).get('content', None)
            return content
        elif response.status_code == 202:
            response_json = response.json()
            for _ in range(5):
                content = self._wait_for_fullfill(response_json.get('reqId'), headers)
                if content is not None:
                    return content
                
                time.sleep(.5)
            
            raise Exception(f"Failed to get response in time: {response} - {response.json()}")
        else:
            raise Exception(f"Failed to get response: {response} - {response.json()}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint": self.endpoint}