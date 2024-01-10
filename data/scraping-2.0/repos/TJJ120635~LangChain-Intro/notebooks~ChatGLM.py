from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModel, AutoConfig

import torch


def torch_gc():
    # with torch.cuda.device(DEVICE):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    tokenizer: object = None
    model: object = None

    def __init__(
        self,
        model_path: str = "chatglm-6b-int4",
        **kwargs
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True, 
            revision=model_path
        )
        model_config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            revision=model_path
        )
        self.model = AutoModel.from_pretrained(
            model_path, 
            config=model_config, 
            trust_remote_code=True, 
            revision=model_path, 
            **kwargs
        ).half().cuda()
        self.model = self.model.eval()

    def __del__(self):
        self.tokenizer = None
        self.model = None

        torch.cuda.empty_cache()
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(
        self, 
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        return response