from dataclasses import dataclass, field
from typing import Optional

import openai


@dataclass
class ChatSession:
    history: list = field(default_factory=list)

    def user_says(self, message):
        self.history.append({"role": "user", "content": message})

    def system_says(self, message):
        self.history.append({"role": "system", "content": message})

    def assistant_says(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_ai_response(
        self,
        initial_system_msg: Optional[str] = None,
        reinforcement_system_msg: Optional[str] = None,
    ):
        chat_history = self.history[:]
        # add the initial system message describing the AI's role
        if initial_system_msg:
            chat_history.insert(0, {"role": "system", "content": initial_system_msg})

        if reinforcement_system_msg:
            chat_history.append({"role": "system", "content": reinforcement_system_msg})

        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", messages=chat_history, timeout=15
        )


class FlaggedInputError(RuntimeError):
    pass


def check_for_flagged_content(msg: str):
    response = openai.Moderation.create(msg)
    if response.results[0].flagged:
        raise FlaggedInputError()
