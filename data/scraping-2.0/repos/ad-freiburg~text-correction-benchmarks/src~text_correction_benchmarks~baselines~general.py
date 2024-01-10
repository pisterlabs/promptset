import os
from typing import Iterable, Any

from text_correction_benchmarks.baselines import Baseline
from text_utils import text

import openai
from openai.error import OpenAIError, RateLimitError


class ChatGPT(Baseline):
    def __init__(self, task: str):
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY env variable must be set"
        if task == "wsc":
            self.system_prompt = "You are a helpful assistant that fixes " \
                "whitespace errors, but not spelling errors in text. " \
                "You only ever respond with the corrected text " \
                "and no additional information. If there are no errors, you respond with the original text."
        elif task == "seds":
            self.system_prompt = "You are a helpful assistant that detects " \
                "whether a text contains a spelling error. You respond with ERROR " \
                "if there is a spelling error, and with NO_ERROR if there is none. " \
                "You do not add any additional information to your response."
        elif task == "sec":
            self.system_prompt = "You are a helpful assistant that fixes " \
                "spelling errors in text. You only ever respond with the corrected text " \
                "and no additional information. If there are no errors, you respond with the original text."
        elif task == "gec":
            self.system_prompt = "You are a helpful assistant that fixes " \
                "spelling errors and grammatical errors in text. You only ever respond with the corrected text " \
                "and no additional information. If there are no errors, you respond with the original text."
        else:
            raise ValueError(f"unsupported task {task} for ChatGPT baseline")

    def run(self, sequences: Iterable[str], **kwargs: Any) -> Iterable[str]:
        for sequence in sequences:
            num_tries = 0
            while num_tries < 3:
                try:
                    result = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": sequence}
                        ]
                    )
                except RateLimitError:
                    continue
                except OpenAIError:
                    num_tries += 1
                    continue

                yield text.clean(result["choices"][0]["message"]["content"])
                break

    @property
    def name(self) -> str:
        return "ChatGPT"
