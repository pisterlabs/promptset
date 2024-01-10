
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping
# import logging

# logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
# consoleHandler = logging.StreamHandler()
# consoleHandler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# consoleHandler.setFormatter(formatter)

# logger.addHandler(consoleHandler)

class Baichuan(LLM):
  
    @property
    def _llm_type(self) -> str:
        return super()._llm_type
    
    def _call(self, prompt: str, stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> dict:
       
        import json 
        import requests
        url = "http://127.0.0.1:8000/"
        headers = {
            'Content-Type': 'application/json;charset=utf-8'
        }
        payload = json.dumps({
        "role": "user",
        "content": prompt
        })
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = requests.request("POST", url=url, headers=headers, data=payload)

        return response.json()
    

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return super()._identifying_params
