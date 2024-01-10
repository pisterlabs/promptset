from dataclasses import dataclass, field
from typing import Optional
from typing import TypedDict
from logzero import logger
import openai


@dataclass
class ChatSession:
    initial_system_message: Optional[str] = (None,)
    reinforcement_system_msg: Optional[str] = None
    history: list = field(default_factory=list)
    model: str = "gpt-3.5-turbo-0613"

    def user_says(self, message):
        self.history.append({"role": "user", "content": message})

    def system_says(self, message):
        self.history.append({"role": "system", "content": message})

    def assistant_says(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_ai_response(self, initial_system_msg: Optional[str] = None, reinforcement_system_msg: Optional[str] = None):
        initial_system_msg = initial_system_msg or self.initial_system_message
        reinforcement_system_msg = reinforcement_system_msg or self.reinforcement_system_msg

        chat_history = self.history[:]
        # add the initial system message describing the AI's role
        if initial_system_msg:
            chat_history.insert(0, {"role": "system", "content": initial_system_msg})

        if reinforcement_system_msg:
            chat_history.append({"role": "system", "content": reinforcement_system_msg})
        logger.info('Generating AI ChatCompletion')
        logger.debug(chat_history)
        response = openai.ChatCompletion.create(model=self.model, messages=chat_history)
        logger.debug(response)
        return response


class FlaggedInputError(RuntimeError):
    pass


def check_for_flagged_content(msg: str):
    response = openai.Moderation.create(msg)
    if response.results[0].flagged:
        raise FlaggedInputError()
