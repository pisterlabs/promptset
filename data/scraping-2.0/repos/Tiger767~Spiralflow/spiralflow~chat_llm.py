from typing import Dict, List, Tuple
import openai
import backoff

from .message import Message


class ChatLLM:
    """
    A class for chat completion using the GPT model.
    """

    def __init__(
        self, gpt_model: str = "gpt-3.5-turbo", stream_hook=None, **kwargs
    ) -> None:
        """
        Initializes the ChatLLM class with the given parameters.

        :param gpt_model: GPT model to use for chat completion.
        :param stream_hook: A function callled each token recieved via streaming the response.
        """
        self.gpt_model = gpt_model
        self.model_params = kwargs
        self.stream_hook = stream_hook

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
    def __call__(self, messages: List[Message]) -> Tuple[str, str, Dict]:
        """
        Generates a response using the GPT model based on the input messages.

        :param messages: List of messages to use for chat completion.
        :return: Response from the chat completion with content, role, and metadata.
        """
        if self.stream_hook is not None:
            response = openai.ChatCompletion.create(
                model=self.gpt_model, messages=messages, stream=True, **self.model_params
            )
            role = None
            full_content = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "role" in delta:
                    role = delta["role"]

                if "content" in delta:
                    full_content += delta["content"]
                    if self.stream_hook(delta["content"], role, chunk) is False:
                        break
                elif self.stream_hook(None, role, chunk) is False:
                    break
            return full_content, role, chunk
        else:
            response = openai.ChatCompletion.create(
                model=self.gpt_model, messages=messages, **self.model_params
            )
            message = response["choices"][0]["message"]

            return message["content"], message["role"], response
