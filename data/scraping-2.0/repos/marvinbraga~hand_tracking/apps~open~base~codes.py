from enum import Enum

import openai

from apps.open.base.abstract import AbstractOpenAi
from apps.open.base.utils import StopWatch


class CodeEnum(Enum):
    cushman_001 = "code-cushman-001"
    davinci_002 = "code-davinci-002"
    davinci_edit_001 = "code-davinci-edit-001"
    search_ada_code_001 = "code-search-ada-code-001"
    search_ada_text_001 = "code-search-ada-text-001"
    search_babbage_code_001 = "code-search-babbage-code-001"
    search_babbage_text_001 = "code-search-babbage-text-001"


class CodeCreate(AbstractOpenAi):
    def __init__(
        self,
        prompt,
        engine: CodeEnum = CodeEnum.davinci_002,
        language="python/3.11",
        **kwargs,
    ):
        super().__init__(prompt)
        self._prompt += f" Utilize a linguagem/versão '{language}'."
        self._engine = engine
        self._hparams = {
            "max_tokens": 1024,
            "n": 1,
            "stop": None,
            "temperature": 0.5,
        }
        self._hparams.update(**kwargs)

    def execute(self):
        stopwatch = StopWatch()
        try:
            response = openai.Completion.create(
                engine=self._engine.value,
                prompt=self._prompt,
                **self._hparams,
            )
        finally:
            stopwatch.stop()

        self._result = response.choices[0].text.strip()
        return self
