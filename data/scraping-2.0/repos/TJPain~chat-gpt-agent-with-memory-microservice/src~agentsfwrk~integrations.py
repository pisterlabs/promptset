import json
import os
import time
from typing import Union

import openai
from openai.error import APIConnectionError, APIError, RateLimitError

from agentsfwrk.exceptions import (
    InvalidInputError,
    APIResponseError,
    MaxRetriesExceededError,
    InvalidAPIKeyError,
    InvalidContextOrInstructionError,
)
import agentsfwrk.logger as logger

log = logger.get_logger(__name__)

is_testing = os.environ.get("IS_TESTING", "False") == "True"

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key and not is_testing:
    raise InvalidAPIKeyError("API key is missing or invalid.")

MAX_RETRIES = 3
DEFAULT_RETRY_TIME = 3


class OpenAIIntegrationService:
    def __init__(
        self, context: Union[str, dict], instruction: Union[str, dict]
    ) -> None:
        self.context = context
        self.instructions = instruction
        self.messages = []

        if isinstance(self.context, dict):
            self.messages.append(self.context)
        elif isinstance(self.context, str):
            self.messages.append({"role": "system", "content": self.context})
        else:
            raise InvalidContextOrInstructionError(
                "Invalid context or instruction type."
            )

    def get_models(self) -> dict:
        return openai.Model.list()

    def add_chat_history(self, messages: list[dict]) -> None:
        """
        Adds chat history to the conversation.
        """
        if not isinstance(messages, list) or not all(
            isinstance(item, dict) for item in messages
        ):
            raise InvalidInputError("Input must be a list of dictionaries.")
        self.messages += messages

    def _handle_api_exceptions(self, e: Exception, attempt: int) -> int:
        """
        Handle exceptions from the OpenAI API.
        """
        if attempt == MAX_RETRIES - 1:
            log.error(f"Last attempt failed, Exception occurred: {e}.")
            raise MaxRetriesExceededError("Maximum number of retries reached.")
        retry_time = getattr(e, "retry_after", DEFAULT_RETRY_TIME)
        log.error(f"Exception occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return retry_time

    def answer_to_prompt(self, model: str, prompt: str, **kwargs) -> dict[str, str]:
        """
        Collects prompts from user, appends to messages from the same conversation
        and return responses from the gpt models.
        """
        # Preserve the messages in the conversation
        self.messages.append({"role": "user", "content": prompt})
        retry_exceptions = (APIError, APIConnectionError, RateLimitError)

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=model, messages=self.messages, **kwargs
                )
                break
            except retry_exceptions as e:
                self._handle_api_exceptions(e, attempt)

        if response is None:
            log.error("Failed to get a response after all retries.")
            raise APIResponseError("Failed to get a response after all retries.")

        if not response.choices:
            log.error("Received empty choices from the API.")
            raise APIResponseError("Received empty choices from the API.")

        response_message = response.choices[0].message["content"]
        response_data = {"answer": response_message}
        self.messages.append({"role": "assistant", "content": response_message})

        return response_data

    def answer_to_simple_prompt(
        self, model: str, prompt: str, **kwargs
    ) -> dict[str, Union[bool, str]]:
        """
        Collects context and appends a prompt from a user and returns a response from
        the GPT model given an instruction.
        This method only allows one message exchange.
        """

        # Copy the existing messages and append the new user prompt
        messages = self.messages.copy()
        messages.append({"role": "user", "content": prompt})

        retry_exceptions = (APIError, APIConnectionError, RateLimitError)

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.Completion.create(
                    model=model, prompt=json.dumps(messages), **kwargs
                )
                break
            except retry_exceptions as e:
                self._handle_api_exceptions(e, attempt)

        if response is None:
            log.error("Failed to get a response after all retries.")
            raise APIResponseError("Failed to get a response after all retries.")

        response_message = response.choices[0].text

        try:
            response_data = json.loads(response_message)
            answer_text = response_data.get("answer")
            if answer_text is not None:
                messages.append({"role": "assistant", "content": answer_text})
                self.messages = messages
            else:
                raise APIResponseError("The response from the model is not valid.")
        except ValueError as e:
            log.error(f"Error occurred while parsing response: {e}")
            log.error(f"Prompt from the user: {prompt}")
            log.error(f"Response from the model: {response_message}")
            log.info("Returning a safe response to the user.")
            response_data = {"intent": False, "answer": response_message}

        return response_data

    def verify_end_conversation(self) -> None:
        """
        Verify if the conversation has ended by checking the last message from the user
        and the last message from the assistant.
        """
        pass

    def verify_goal_conversation(self, model: str, **kwargs) -> dict:
        """
        Verify if the conversation has reached the goal by checking the conversation history.
        Format the response as specified in the instructions.
        """
        messages = self.messages.copy()
        messages.append(self.instructions)

        retry_exceptions = (APIError, APIConnectionError, RateLimitError)

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=model, messages=messages, **kwargs
                )
                break
            except retry_exceptions as e:
                self._handle_api_exceptions(e, attempt)

        if response is None:
            log.error("Failed to get a response after all retries.")
            raise APIResponseError("Failed to get a response after all retries.")

        response_message = response.choices[0].message["content"]
        try:
            response_data = json.loads(response_message)
            if response_data.get("summary") is None:
                raise APIResponseError(
                    "The response from the model is not valid. Missing summary."
                )
        except ValueError as e:
            log.error(f"Error occurred while parsing response: {e}")
            log.error(f"Response from the model: {response_message}")
            log.info("Returning a safe response to the user.")
            raise APIResponseError("Error occurred while parsing response.")

        return response_data
