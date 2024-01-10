from typing import List
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class ChatGLM(LLM):
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM-6b"

    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model.eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(self.tokenizer,
                                      prompt,
                                      top_p=0.7,
                                      temperature=0.5)
        if stop is None:
            content = response
        else:
            content = enforce_stop_tokens(response, stop)
        return content
