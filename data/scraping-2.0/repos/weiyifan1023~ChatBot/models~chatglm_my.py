import torch
import json
import argparse
import tqdm
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, AutoModel, GenerationConfig
from configs.model_config import *
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.base import LLM


DEVICE = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatGLM(LLM):
    def __int__(self):
        super().__init__()

        self.max_token: int = 10000
        self.temperature: float = 0.01
        self.n = 10  # completion length
        self.top_p = 0.9
        self.history = []
        self.history_len: int = -1  # hotpot不需要history
        self.tokenizer: object = None
        self.model: object = None
        # self.tokenizer = None
        # self.model = None

    @property
    def _llm_type(self) -> str:
        return "My ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)  # Cut off the text as soon as any stop words occur
        self.history = self.history + [[None, response]]  # history 包含了之前所有的query和answer
        return response

    def load_model(self, model_name_or_path: str = "THUDM/chatglm-6b", llm_device=LLM_DEVICE):

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # model
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True
                ).half().cuda()
            )
        else:
            self.model = (
                AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float().to(llm_device)
            )

        self.model = self.model.eval()
        return self.tokenizer, self.model
