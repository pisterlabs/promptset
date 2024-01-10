"""GPT skill."""

import logging
from typing import Optional, Dict

import openai

from searchmate.skill import Skill
from searchmate.config import app_config


class GPTSkill(Skill):
    """
    GPT skill, makes chatting with GPT available in SearchMate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.keywords = ["gpt"]

        self._api_key = app_config.get("gptskill", "api_key")

        self._suggestion_message = "Press Enter to continue..."
        self._no_api_key_message = "Please add OpenAI API key to the config."
        self._server_unavailable_message = (
            "OpenAI servers are currently unavailable."
        )

    def run(self, query: str) -> Optional[Dict[str, str]]:
        """
        Sends request to OpenAI's GPT module.

        Args:
            query: Users' text input.

        Returns:
            Optional[Dict[str, str]]: GPT's response.
        """
        if not self._api_key or self._api_key.isspace():
            return {
                "widget_type": "plain",
                "message": self._no_api_key_message,
            }

        openai.api_key = self._api_key

        logging.debug("GPTSkill - sending request to OpenAI...")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": query},
                ],
            )
        except openai.error.ServiceUnavailableError:
            return {
                "widget_type": "plain",
                "message": self._server_unavailable_message,
            }

        result = response.choices[0].message.content

        return {
            "widget_type": "markdown",
            "message": result,
        }

    def suggestion(self, query: str) -> Optional[Dict[str, str]]:
        """
        What to display before executing skill.

        Args:
            query: Users' text input.

        Returns:
            Optional[Dict[str, str]]: Text to display before skill runs.
        """
        if not query or query.isspace():
            return None

        return {
            "widget_type": "plain",
            "message": self._suggestion_message,
        }
