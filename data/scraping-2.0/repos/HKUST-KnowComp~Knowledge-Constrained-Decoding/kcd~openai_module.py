from dataclasses import dataclass
from typing import Union
import os
import openai


@dataclass
class OpenAIAPIParameters:
    max_tokens: int = 16
    temperature: float = 1  # between (0, 2)
    top_p: float = 1
    n: int = 1  # number of completions for each prompt)
    logprobs: int = 0  # max 5, only for Completion API
    stop: Union[str, list[str]] = None  # stop token, max 4
    best_of: int = 1
    logit_bias: dict[str, int] = None
    presence_penalty: float = 0  # between (-2, 2),
    frequency_penalty: float = 0  # between (-2, 2),

    def __post_init__(self):
        if self.logit_bias is None:
            self.logit_bias = dict()


class OpenAIModel:

    def __init__(self, model_name: str = "text-davinci-003", task: str = 'completion'):
        self.model_name = model_name
        self.openai = openai
        self.openai.organization = os.getenv("OPENAI_ORG_ID")
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        if self.model_name == 'gpt-3.5-turbo':
            self.task = 'chat'
        else:
            self.task = task

    def __call__(self,
                 prompt: str | list[dict[str, str]],
                 parameters: OpenAIAPIParameters,
                 suffix: str = None):
        return self.get_response(prompt, parameters, suffix=suffix)

    def get_response(self,
                     prompt: str | list[dict[str, str]],
                     parameters: OpenAIAPIParameters,
                     suffix: str = None):
        if self.task == 'completion':
            response = self.openai.Completion.create(prompt=prompt,
                                                     model=self.model_name,
                                                     **parameters.__dict__,
                                                     suffix=suffix)
            text = response['choices'][0]['text']
        else:  # chat
            assert isinstance(prompt, list)
            kwargs = parameters.__dict__
            # not used for chat API
            kwargs.pop('logprobs', None)
            kwargs.pop('best_of', None)
            response = self.openai.ChatCompletion.create(messages=prompt,
                                                         model=self.model_name,
                                                         **kwargs)
            text = response['choices'][0]['message']['content']
        return text, response


class MockOpenAIModel:

    def __init__(self, model_name: str = "text-davinci-003", task: str = 'completion') -> None:
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model_name = model_name
        self.task = task

    def __call__(self,
                 prompt: str | list[dict[str, str]],
                 parameters: OpenAIAPIParameters,
                 suffix: str = None):
        return self.get_response(prompt, parameters, suffix=suffix)

    def get_response(self,
                     prompt: str | list[dict[str, str]],
                     parameters: OpenAIAPIParameters,
                     suffix: str = None):
        mock_tokens = self.tokenizer.convert_ids_to_tokens([1, 2, 3, 4, 5])
        mock_logprobs = [-1, -2, -3, -4, -5]
        n_prompt_tokens = len(self.tokenizer.encode(prompt))
        total_usage = n_prompt_tokens + 1  # 1 for generate 1 token
        mock_response = {
            'choices': [{
                'text': 'mock response',
                'logprobs': {
                    'tokens': ['!'],
                    'token_logprobs': [-0.9],
                    'top_logprobs': [dict(zip(mock_tokens, mock_logprobs))]
                }
            }],
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": total_usage
            }
        }
        return 'mock response', mock_response