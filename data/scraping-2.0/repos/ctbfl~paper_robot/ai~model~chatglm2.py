from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoTokenizer, AutoModel

from .utils import find_substrings

# # 服务器版本
# # LLM_location = "/data/pretrained_models/chatglm2-6b"

# # 本地版本
# LLM_location = r"D:\LLMs\chatglm2-6b"


# tokenizer = AutoTokenizer.from_pretrained(LLM_location, trust_remote_code=True)
# model = AutoModel.from_pretrained(LLM_location, trust_remote_code=True, device='cuda')
# model = model.eval()




class ChatGLM2(LLM):
    tokenizer: object = None
    model: object = None

    @property
    def _llm_type(self) -> str:
        return "ChatGLM2"

    def load_model(self, model_name_or_path = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device='cuda')
        self.model = self.model.eval()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history=[])

        if stop is not None:
            response = find_substrings(response, stop)

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": 6}