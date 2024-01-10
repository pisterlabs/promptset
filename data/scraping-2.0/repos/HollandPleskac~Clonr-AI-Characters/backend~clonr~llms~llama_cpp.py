from clonr.tokenizer import Tokenizer

from .callbacks import LLMCallback
from .openai import OpenAI


class LlamaCpp(OpenAI):
    model_type = "llama-cpp"
    is_chat_model: bool = True

    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_base: str = "http://localhost:8100/v1",
        chat_mode: bool = True,
        tokenizer: Tokenizer | None = None,
        callbacks: list[LLMCallback] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.chat_mode = chat_mode
        # TODO fix this so it always aligns?
        self.tokenizer = tokenizer or Tokenizer.from_huggingface(
            "TheBloke/Llama-2-13B-chat-GPTQ"
        )
        self.callbacks = callbacks or []

    @property
    def default_system_prompt(self):
        return "You are a helpful assistant."

    @property
    def context_length(self) -> int:
        return 4096  # llama2 context window
