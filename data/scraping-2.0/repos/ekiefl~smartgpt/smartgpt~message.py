from __future__ import annotations

import time
from typing import Dict, List

import attrs
import cattrs
import openai
from openai.error import APIError, InvalidRequestError, RateLimitError

from smartgpt import strenum
from smartgpt.logger import default_logger
from smartgpt.settings.credentials import Credentials


class Role(strenum.StrEnum):
    """Enum representing the different roles in a conversation"""

    USER = strenum.auto()
    ASSISTANT = strenum.auto()
    SYSTEM = strenum.auto()


@attrs.define(frozen=True)
class Message:
    """Structure for encapsulating a message in a conversation.

    Attributes:
        role:
            The role of the sender of the message.
        content:
            The content of the message.
    """

    role: Role
    content: str


@attrs.define
class Response:
    """Structure for encapsulating a response from the GPT model.

    Attributes:
        message:
            The message from the model.
        total_tokens:
            The total number of tokens used.
        finish_reason:
            The reason for finishing the response.
    """

    message: Message
    total_tokens: int
    finish_reason: str

    @classmethod
    def from_openai_response(cls, response) -> Response:
        """Factory method to create a Response from an OpenAI API response.

        Args:
            response:
                The response from the OpenAI API.

        Returns:
            Response:
                A Response instance with values from the API response.
        """
        return cls(
            message=cattrs.structure(response["choices"][0]["message"], Message),
            total_tokens=response["usage"]["total_tokens"],
            finish_reason=response["choices"][0]["finish_reason"],
        )


@attrs.define
class GPTBot:
    """Represents an interface to the GPT model.

    It encapsulates the process of sending messages to the model and receiving
    responses. The class also handles message history.

    Attributes:
        messages:
            A list of messages sent to and received from the model.
        credentials:
            Credentials for accessing the model.
        model:
            The model to use (default is 'gpt-4').
        temp:
            The temperature parameter to use when generating responses.
    """

    messages: List[Dict[str, str]] = attrs.field(factory=list)
    credentials: Credentials = attrs.field(default=Credentials.default())
    model: str = attrs.field(default="gpt-4")
    temp: float = attrs.field(default=0.5)

    def append_message(self, message: Message) -> None:
        """Appends a message to the current message history.

        Args:
            message: The message to append.
        """
        self.messages.append(attrs.asdict(message))

    def request(self) -> Response:
        """Sends the current message history to the GPT model via an API request

        The message history includes all previous interactions, which allows the model
        to generate a response based on the entire conversation context.

        Returns:
            Response:
                A Response object that encapsulates the model's response, which includes
                the generated message, remaining tokens, and other metadata.
        """

        try:
            return Response.from_openai_response(
                openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages,
                    api_key=self.credentials.key,
                )
            )
        except RateLimitError:
            default_logger.info("Hit rate limit. Sleeping for 20 seconds...")
            time.sleep(20)
            return self.request()
        except APIError:
            default_logger.info("Potentially bad gateway. Sleeping for 20 seconds...")
            time.sleep(20)
            return self.request()
        except InvalidRequestError:
            raise NotImplementedError()

    def response(self, prompt: str) -> Message:
        """Appends prompt to message history and sends request to the GPT model.

        The model's response is then appended to the message history.

        Args:
            prompt:
                The prompt to send to the model.

        Returns:
            Message:
                The model's response encapsulated in a Message object.
        """
        self.append_message(Message(Role.USER, prompt))

        response = self.request()

        self.append_message(response.message)

        return response.message
