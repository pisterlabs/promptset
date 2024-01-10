from anthropic import Anthropic as AnthropicClient
from anthropic import AI_PROMPT

from services.llm.LLM import LLM

class Anthropic(LLM):
    def __init__(self):
        super().__init__()

    def get_models(self):
        return ['claude-2.1']

    def generate(self, prompt, model):
        client = AnthropicClient()

        completion = client.completions.create(
            model=model,
            max_tokens_to_sample=2048,
            prompt=prompt,
        )

        return completion.completion
    

