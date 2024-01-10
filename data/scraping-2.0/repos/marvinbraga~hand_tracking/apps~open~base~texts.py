import openai

from apps.open.base.abstract import AbstractOpenAi, TextEngineEnum
from apps.open.base.utils import StopWatch


class TextCreate(AbstractOpenAi):
    engine = TextEngineEnum.DAVINCI_003.value

    def __init__(self, prompt, **kwargs):
        super().__init__(prompt)
        self._hparams = {
            "temperature": 0.7,
            "max_tokens": 256 * 6,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self._hparams.update(**kwargs)

    def execute(self):
        stopwatch = StopWatch()
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=self._prompt,
                **self._hparams,
            )
        finally:
            stopwatch.stop()

        self._result = response.choices[0].text
        return self
