import enum

import numpy as np
import openai
import tiktoken
from openai.error import AuthenticationError, InvalidRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type


class TokenCounter:
    def __init__(self, model, texts, indicate_shared=False):
        enc = tiktoken.encoding_for_model(model)
        self.is_single = isinstance(texts, str)
        if self.is_single:
            texts = [texts]
        full_text = "".join(texts)
        self.ends = np.cumsum(
            [len(e) for e in enc.decode_batch([e] for e in enc.encode(full_text))]
        )
        self.num_all_tokens = len(self.ends)
        self.num_non_special_tokens = len(self.ends)
        self.num_special_tokens = self.num_all_tokens - self.num_non_special_tokens
        self.counts = []
        self.current_count = 0
        self.current_length = 0
        self.length_iter = iter(np.cumsum([len(e) for e in texts]))
        self.indicate_shared = indicate_shared

    def results(self):
        return {
            "counts": self.counts,
            "num": {
                "all": self.num_all_tokens,
                "special": self.num_special_tokens,
                "non_special": self.num_non_special_tokens,
            },
        }

    def _commit_count(self, is_partial):
        if self.indicate_shared and is_partial:
            self.current_count += 0.5
        self.counts.append(self.current_count)
        self.current_count = 0

    def _get_next_length(self, is_partial):
        while (next_length := next(self.length_iter)) == self.current_length:
            self._commit_count(is_partial)
        return next_length

    def consume(self):
        if self.counts:
            raise Exception("already consumed")
        try:
            self.current_length = self._get_next_length(False)
            for end in self.ends:
                self.current_count += 1
                while self.current_length <= end:
                    self._commit_count(self.current_length != end)
                    self.current_length = self._get_next_length(
                        self.current_length != end
                    )
        except StopIteration:
            pass
        if self.is_single:
            (self.counts,) = self.counts
        return self.counts


class MODEL_TYPES(enum.Enum):
    COMPLETION = enum.auto()
    CHAT = enum.auto()


MODELS = {
    "text-davinci-003": {"max_length": 4096, "type": MODEL_TYPES.COMPLETION},
    "text-davinci-002": {"max_length": 4096, "type": MODEL_TYPES.COMPLETION},
    "text-curie-001": {"max_length": 2048, "type": MODEL_TYPES.COMPLETION},
    "text-babbage-001": {"max_length": 2048, "type": MODEL_TYPES.COMPLETION},
    "text-ada-001": {"max_length": 2048, "type": MODEL_TYPES.COMPLETION},
    "gpt-3.5-turbo": {"max_length": 4096, "type": MODEL_TYPES.CHAT},
    "gpt-4": {"max_length": 8192, "type": MODEL_TYPES.CHAT},
    "gpt-4-32k": {"max_length": 32768, "type": MODEL_TYPES.CHAT},
}


class OpenAIClient:
    MODELS = set(MODELS.keys())

    def __init__(self, model, api_key):
        self.model = model
        if api_key is None:
            raise ValueError("api_key is None")
        self.api_key = api_key
        model_info = MODELS[model]
        self.model_max_length = model_info["max_length"]
        self.is_chat = model_info["type"] == MODEL_TYPES.CHAT

    def generate(self, prompt, system_message=None, **kwargs):
        if self.is_chat:
            if system_message is None:
                raise ValueError("system_message is None with a Chat model")
            result = openai.ChatCompletion.create(
                model=self.model,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                **kwargs,
            )
        else:
            if system_message is not None:
                raise ValueError("system_message is not None with a Completion model")
            result = openai.Completion.create(
                model=self.model,
                api_key=self.api_key,
                prompt=prompt,
                **kwargs,
            )
        usage = result.usage
        (choice,) = result.choices
        text = choice.text if "text" in choice else choice.message.content
        return {
            "generated": text,
            "size": {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "overflow": 0,
            },
            "stopping_reason": choice.finish_reason,
        }

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
        retry=retry_if_not_exception_type(
            (
                AuthenticationError,
                NotImplementedError,
                KeyboardInterrupt,
                InvalidRequestError,
                ValueError,
            )
        ),
    )
    def _generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def meta(self):
        return {
            "model": self.model,
            "model_max_length": self.model_max_length,
            "architecture_type": "decoder",
        }

    def count_tokens(self, text, indicate_shared=False):
        counter = TokenCounter(self.model, text, indicate_shared=indicate_shared)
        counter.consume()
        return counter.results()

    def __call__(
        self,
        batch,
        max_new_tokens=None,
        with_meta=False,
        temperature=0,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        is_single = not isinstance(batch, list)
        if is_single:
            batch = [batch]
        generated = []
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        for e in batch:
            if isinstance(e, str):
                result = self._generate(prompt=e, **kwargs)
            elif isinstance(e, (list, tuple)):
                try:
                    system_message, prompt = e
                except:
                    (prompt,) = e
                result = self._generate(
                    prompt=prompt, system_message=system_message, **kwargs
                )
            elif isinstance(e, dict):
                result = self._generate(**e, **kwargs)
            else:
                raise ValueError("input has to be on of [str, list, tuple, dict]")
            generated.append(result)
        if is_single:
            (generated,) = generated
        if with_meta:
            return generated, self.meta()
        return generated
