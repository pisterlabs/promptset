
from typing import Any, Mapping
import openai

from utils import dotdict

class TextModel(object):
    def __init__(self, model_name='text-davinci-003', **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 300)
        self.n = kwargs.get('n', 1)

    def _create_text_result(self, response: Mapping[str, Any]):
        generations = []
        for res in response["choices"]:
            generations.append(res["text"])
        llm_output = { "token_usage": response["usage"], 'model_name': self.model_name, 'generations': generations }
        return dotdict(llm_output)

    def __call__(self, prompt, stop=None):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=stop
        )
        return self._create_text_result(response)
