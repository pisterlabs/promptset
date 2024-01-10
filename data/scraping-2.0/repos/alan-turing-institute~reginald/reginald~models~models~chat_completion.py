import logging
import os
from typing import Any

import openai

from reginald.models.models.base import MessageResponse, ResponseModel
from reginald.utils import get_env_var


class ChatCompletionBase(ResponseModel):
    def __init__(self, *args, **kwargs) -> None:
        """
        Base class for chat completion models.
        """
        super().__init__(emoji="books")


class ChatCompletionAzure(ChatCompletionBase):
    def __init__(
        self, model_name: str = "reginald-curie", *args: Any, **kwargs: Any
    ) -> None:
        """
        Simple chat completion model using Azure's
        instance of OpenAI's LLMs to implement the LLM.

        Must have the following environment variables set:
        - `OPENAI_API_BASE`: Azure endpoint which looks
          like https://YOUR_RESOURCE_NAME.openai.azure.com/
        - `OPENAI_API_KEY`: Azure API key

        Parameters
        ----------
        model_name : str, optional
            Deployment name of the model on Azure, by default "reginald-curie"
        """
        logging.info(f"Setting up AzureOpenAI LLM (model {model_name})")
        super().__init__(*args, **kwargs)
        self.api_base = get_env_var("OPENAI_AZURE_API_BASE", secret_value=False)
        self.api_key = get_env_var("OPENAI_AZURE_API_KEY")
        self.api_type = "azure"
        self.api_version = "2023-03-15-preview"
        self.best_of = 1
        self.engine = model_name  # the deployment name
        self.frequency_penalty = 0
        self.max_tokens = 100
        self.presence_penalty = 0
        self.temperature = 0.2
        self.top_p = 0.95

    def _respond(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a message in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        openai.api_base = self.api_base
        openai.api_type = self.api_type
        openai.api_version = self.api_version
        openai.api_key = self.api_key
        response = openai.Completion.create(
            best_of=self.best_of,
            engine=self.engine,
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            prompt=message,
            stop=None,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return MessageResponse(response["choices"][0]["text"])

    def direct_message(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a direct message in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        return self._respond(message=message, user_id=user_id)

    def channel_mention(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a channel mention in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        return self._respond(message=message, user_id=user_id)


class ChatCompletionOpenAI(ChatCompletionBase):
    def __init__(
        self, model_name: str = "gpt-3.5-turbo", *args: Any, **kwargs: Any
    ) -> None:
        """
        Simple chat completion model using OpenAI's API.

        Must have `OPENAI_API_KEY` set as an environment variable.

        Parameters
        ----------
        model_name : str, optional
            Model name on OpenAI, by default "gpt-3.5-turbo"
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.api_key = get_env_var("OPENAI_API_KEY")

    def _respond(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a message in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return MessageResponse(response["choices"][0]["message"]["content"])

    def direct_message(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a direct message in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        return self._respond(message=message, user_id=user_id)

    def channel_mention(self, message: str, user_id: str) -> MessageResponse:
        """
        Method to respond to a channel mention in Slack.

        Parameters
        ----------
        msg_in : str
            Message from user
        user_id : str
            User ID

        Returns
        -------
        MessageResponse
            Response from the query engine.
        """
        return self._respond(message=message, user_id=user_id)
