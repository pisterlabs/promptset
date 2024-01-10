"""
LLM classes
"""
import os
from typing import Optional

import openai


class PromptBase:
    """
    Base class for creating Prompts
    """

    def as_dict(self) -> dict:
        """
        Get prompt as kwargs dict for library call
        """
        raise NotImplementedError()

    def get_llm_response(self):
        """
        Get response to the prompt from LLM
        """
        raise NotImplementedError()


class OpenAIPrompt(PromptBase):
    """
    Prompt for OpenAI models
    """

    def __init__(
        self,
        model_settings: dict,
        messages: Optional[list[dict]] = None,
        system_message: Optional[str] = None,
    ):
        self._model_settings = model_settings
        self._messages = messages or []

        if system_message:
            self.add_message("system", system_message)

    def add_message(self, role: str, content: str):
        """
        Add message to prompt messages
        :param role: role of the message
        :param content: Content of the message
        """
        self._messages.append(({"role": role, "content": content}))

    def as_dict(self) -> dict:
        return {
            **self._model_settings,
            "messages": self._messages,
        }

    def get_llm_response(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        return openai.ChatCompletion.create(**self.as_dict())
