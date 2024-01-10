from copy import deepcopy
from logging import getLogger
from os import getenv
from typing import Optional, Callable

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, BadRequestError

from ..chat import Chat

log = getLogger(__name__)


class Claude:
    MODELS = [
        "claude-2",
        "claude-instant-1",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "claude-2",
    ):
        if api_key is None:
            api_key = getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Antrophic API key must be provided")

        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}")

        self.api_key = api_key
        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    @staticmethod
    def _sanitize_message(txt: str) -> str:
        """
        Sanitize content so there's no prompt in the middle of the message.

        After each iteration, the function rechecks whether the prompt is still
        in the message and loops if it is. This is necessary to handle cases like:

        "\n\nHu\n\nHuman:man"

        ...which would be sanitized to: "\n\nHuman" the first time, allowing prompt
        injection.
        """
        while True:
            new_txt = txt.replace(HUMAN_PROMPT, "").replace(AI_PROMPT, "")
            if new_txt == txt:
                return new_txt
            txt = new_txt

    @classmethod
    def get_text_prompt(cls, chat: Chat) -> str:
        prompt = HUMAN_PROMPT + " "
        speaker_is_human = True
        for message in chat:
            if message["role"] in ("system", "user"):
                if not speaker_is_human:
                    prompt += HUMAN_PROMPT + " "
                    speaker_is_human = True
            elif message["role"] == "assistant":
                if speaker_is_human:
                    prompt += AI_PROMPT + " "
                    speaker_is_human = False
            else:
                raise ValueError(f"Unsupported message role: {message['role']}")

            prompt += cls._sanitize_message(message["content"]) + "\n"

        # Prime the assistant to reply
        prompt += AI_PROMPT

        # Our newlines + human/assistant prompt newlines are too many, remove the
        # extra ones just before the speaker change.
        prompt = prompt.replace("\n\n\n", "\n\n").rstrip()

        return prompt

    @staticmethod
    def inject_tools(chat: Chat, tools: list | None) -> Chat:
        """
        Inject tool messages into the chat.

        This is necessary because Claude doesn't support tool injection yet.
        """
        if not tools:
            return chat

        new_chat = chat.fork()
        new_chat.messages = new_chat.messages[:1]

        descriptions = "\n".join([f"* {t._oneline_description}" for t in tools])
        new_chat.system(
            """
You have access to the following tools:
%s

To use a tool, return a message in the following JSON format:
{
    "action": "USE_TOOL",
    "name": "<tool-name>",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2",
        ...
    }
}
"""
            % descriptions
        )
        new_chat.messages.extend(deepcopy(chat.messages[1:]))
        return new_chat

    def __call__(
        self,
        chat: Chat,
        max_tokens: int = 1000,
        tools: list | None = None,
        parser: Callable | None = None,
        max_iterations: int = 5,
    ) -> Optional[str]:
        if parser:
            raise NotImplementedError("Custom parsers not implemented yet")

        if tools:
            chat = self.inject_tools(chat, tools)

        prompt = self.get_text_prompt(chat)
        log.debug(f"Calling Claude with prompt: {prompt}")

        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
            )
        except BadRequestError as err:
            log.warning(f"Error calling Claude: {err}", exc_info=True)
            return None

        return response.completion
