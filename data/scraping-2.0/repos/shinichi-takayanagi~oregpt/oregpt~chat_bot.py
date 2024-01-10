import json
from copy import deepcopy

import openai

from oregpt.stdinout import StdInOut


class ChatBot:
    def __init__(self, model: str, assistant_role: str, std_in_out: StdInOut):
        self._std_in_out = std_in_out
        # Model list: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
        # https://platform.openai.com/docs/models/overview
        self._model = model
        self._assistant_role = [{"role": "system", "content": assistant_role}]
        self._initialize_log()

    def _initialize_log(self) -> None:
        self._log: list[dict[str, str]] = deepcopy(self._assistant_role)

    @property
    def model(self) -> str:
        return self._model

    @property
    def assistant_role(self) -> list[dict[str, str]]:
        return deepcopy(self._assistant_role)

    @property
    def log(self) -> list[dict[str, str]]:
        return self._log

    @property
    def std_in_out(self) -> StdInOut:
        return self._std_in_out

    def respond(self, message: str) -> str:
        self._log.append({"role": "user", "content": message})
        with self._std_in_out.print_assistant_thinking():
            # API Reference: https://platform.openai.com/docs/api-reference/completions/create
            response = openai.ChatCompletion.create(
                model=self._model,
                messages=self._log,
                max_tokens=1024,
                temperature=0,
                stream=True,
            )

        content = ""
        for chunk in response:
            chunked_content = chunk["choices"][0]["delta"].get("content", "")
            self._std_in_out.print_assistant(chunked_content)
            content += chunked_content
        self._std_in_out.print_assistant("\n")
        self._log.append({"role": "assistant", "content": content})
        return content

    def save(self, file_name: str) -> str:
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(self._log, file, indent=4, ensure_ascii=False)
        return file_name

    def load(self, file_name: str) -> None:
        with open(file_name, "r", encoding="utf-8") as file:
            self._log = json.load(file)

    def clear(self) -> None:
        self._initialize_log()
