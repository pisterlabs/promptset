import json
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from configs.model_config import LLM_DEVICE
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, Tuple, Union, Optional
from configs.model_config import OPENAI_API_KEY_IRENSHI

DEVICE = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
openai_api_key = OPENAI_API_KEY_IRENSHI

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class ChatGPT(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    streaming: bool = False
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGPT"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              stop: Optional[List[str]] = None) -> str:
        print(prompt)
        print(openai_api_key)

        # 多租户的history怎么处理，建议前端处理
        chat = ChatOpenAI(openai_api_key = openai_api_key, model_name='gpt-3.5-turbo', \
                          temperature=self.temperature, verbose=True)
        # history=history[-self.history_len:] if self.history_len > 0 else []

        response = chat([HumanMessage(content=prompt)])

        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        history = history + [[None, response]]
        return response, history

