from typing import Optional

import instructor
import openai
from pydantic import BaseModel

instructor.patch()


class InputFormat(BaseModel):
    question: str


class OutputFormat(BaseModel):
    answer: str


class BaseApp:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = {}
        if config:
            self.cfg.update(config)

    def __call__(self, app_input: InputFormat) -> OutputFormat:
        return OutputFormat(answer="The answer is always 42.")


class GptBaseApp(BaseApp):
    def __init__(self, config: Optional[dict] = None):
        self.cfg = {
            "gpt_version": "gpt-3.5-turbo-0613",
            "system_prompt": "You are a first-aid expert. Answer the user question. Be accurate and concise.",
        }
        if config:
            self.cfg.update(config)

    def __call__(self, app_input: InputFormat) -> OutputFormat:
        result: OutputFormat = openai.ChatCompletion.create(
            model=self.cfg["gpt_version"],
            response_model=OutputFormat,
            messages=[
                {"role": "system", "content": self.cfg["system_prompt"]},
                {"role": "user", "content": app_input.question},
            ],
        )
        return result


class HumanApp(BaseApp):
    def __call__(self, app_input: InputFormat) -> OutputFormat:
        answer = input(app_input.question)
        return OutputFormat(answer=answer)
