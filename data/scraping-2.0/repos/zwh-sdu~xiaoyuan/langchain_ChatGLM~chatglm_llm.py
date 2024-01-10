from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
import torch
import requests
import json

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatGLM(LLM):

    url: str
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    max_length: int = 4096
    history_len: int = 10

    # def __init__(self):
    #     super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        data = {"input": prompt,
        "temperature": self.temperature,
        "max_length": self.max_length,
        "history": self.history[-self.history_len:],
        "top_p": self.top_p}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        # print("222222222:",response['history'])
        response = response['response']
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history.append([None, response])
        # print("-------->history:",self.history)
        # print("------------------->response:\n",response)
        return response