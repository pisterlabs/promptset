"""
This module contains the ChatApp class for interacting with the OpenAI Chat API.
"""
import os
import logging
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
    AuthenticationError,
    RateLimitError,
    APIError,
)

API_MODEL = os.getenv(
    "API_MODEL", "gpt-3.5-turbo-1106"
)  # Default to 'gpt-3.5-turbo-1106' if not set


class ChatApp:
    """
    The ChatApp class provides methods for setting system messages,
    sending user messages, and receiving assistant messages through the OpenAI Chat API.
    It also includes error handling and logging functionality.
    """

    def __init__(self, **options):
        """
        Initializes a new instance of the ChatApp class.
        Args:
            **options: Additional options to be passed to the OpenAI API.
        """
        self.client = OpenAI()
        self.options = options
        self.messages = []
        logging.info("Chat app initialized with options: %s", self.options)

    def set_system_message(self, system_message: str):
        """
        Sets a system message to be displayed in the chat.
        Args:
            system_message (str): The system message to be displayed.
        """
        self.messages = [{"role": "system", "content": system_message}]

    def chat(self, user_message: str) -> str:
        """
        Sends a user message to the chat and receives an assistant message in response.
        Args:
            user_message (str): The user message to be sent.
        Returns:
            str: The assistant message received in response, or None if an error occurs.
        """
        self.messages.append({"role": "user", "content": user_message})
        try:
            response = self.client.chat.completions.create(
                model=API_MODEL, **self.options, messages=self.messages
            )
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            self._log_chat_session(response)
            return assistant_message
        except (
            APITimeoutError,
            APIConnectionError,
            BadRequestError,
            AuthenticationError,
            PermissionError,
            RateLimitError,
            APIError,
        ) as e:
            logging.error("OpenAI API request failed: %s", e)
            raise

    def _log_chat_session(self, response):
        """
        Logs the details of the chat session.
        Args:
            response: The response object from the OpenAI API.
        """
        logging_object = {
            "session_id": response.id,
            "options": self.options,
            "messages": self.messages,
            "usage": response.usage,
        }
        logging.info(logging_object)
