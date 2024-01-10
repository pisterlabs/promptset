from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import torch
import os
import time

CACHE_DIR = '/mnt/nas2/GrimaRepo/tvergara'


class BaseLLM(LLM):
    model_name: str
    max_new_tokens = 20
    min_new_tokens = 10
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    device: Optional[str] = None

    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.init_models(device)

    def init_models(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)#, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)#, cache_dir=CACHE_DIR, load_in_4bit=True, device_map="auto")
        self.device = device

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **tokens,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens
        )
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response.removeprefix(prompt)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Intel/neural-chat-7b-v3-1"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
    llm = BaseLLM(model_name=model_name)

    initial_time = time.time()
    print(llm("What is the capital of France?"))
    print(llm("What is the capital of Argentina?"))
    print(llm("What is the capital of EEUU?"))
    print(llm("What is the capital of Chile?"))
    print(time.time() - initial_time)
