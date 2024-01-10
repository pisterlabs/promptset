# File to access the API of Together AI for a given LLM
from langchain.llms.base import LLM, BaseLanguageModel
from typing import Optional, List, Mapping, Any
import requests
import json
from collections import deque

class TogetherLLM(LLM):
    bearerToken:str 
    server_url:str = "https://api.together.xyz/inference"
    pload={
        "model": "togethercomputer/llama-2-70b-chat",
        "max_tokens": 86,
        "prompt": "Hi, how are you?",
        "request_type": "language-model-inference",
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "<human>:"
        ],
        "safety_model": "",
        "repetitive_penalty": 1,
        "update_at": "2023-09-11T13:47:21.562Z"
    }
    
    model:str = "togethercomputer/llama-2-70b-chat"
    @property
    def _llm_type(self) -> str:
        
        return "vicuna"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.pload["prompt"] = "### Human:"+prompt+"\n### Assistant:"
        if stop is None or len(stop) == 0:
            stop = ["\n### Human:"]
        #self.pload["stop"] = stop[0]
        self.pload["model"] = self.model
        return self.ask_chatbot_without_stream(self.pload)

      
    def ask_chatbot_without_stream(self, pload):
        headers={"Authorization": f"Bearer {self.bearerToken}"}
        response = requests.post(
            self.server_url,
            json=pload,
            headers=headers
        )
        if response.status_code == 200:
            return(response.json()['output']['choices'][0]['text'].strip())
        else:
            return("Error in generation "+ response.status_code)
