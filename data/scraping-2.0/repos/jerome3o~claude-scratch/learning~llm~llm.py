from abc import ABC, abstractmethod
from anthropic import Anthropic
from anthropic.types import Completion


class Llm(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> bool:
        pass


class AnthropicLlm(Llm):
    def __init__(self, client: Anthropic = None):
        self._client = client or Anthropic()

    def complete(self, prompt: str) -> bool:
        response: Completion = self._client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=300,
            model="claude-2",
        )
        return "{" + response.completion
