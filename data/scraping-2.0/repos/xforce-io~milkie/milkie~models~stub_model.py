import time
from typing import Any, Dict, List, Optional, Union

from openai import Stream

from milkie.messages import OpenAIMessage
from milkie.models import BaseModelBackend
from milkie.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    ModelType,
)

from milkie.utils import BaseTokenCounter


class StubTokenCounter(BaseTokenCounter):

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        r"""Token counting for STUB models, directly returning a constant.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            int: A constant to act as the number of the tokens in the
                messages.
        """
        return 10


class StubModel(BaseModelBackend):
    r"""A dummy model used for unit tests."""
    model_type = ModelType.STUB

    def __init__(self, model_type: ModelType,
                 model_config_dict: Dict[str, Any]) -> None:
        r"""All arguments are unused for the dummy model."""
        super().__init__(model_type, model_config_dict)
        self._token_counter: Optional[BaseTokenCounter] = None

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = StubTokenCounter()
        return self._token_counter

    def run(
        self, messages: List[OpenAIMessage]
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Run fake inference by returning a fixed string.
        All arguments are unused for the dummy model.

        Returns:
            Dict[str, Any]: Response in the OpenAI API format.
        """
        ARBITRARY_STRING = "Lorem Ipsum"
        response: ChatCompletion = ChatCompletion(
            id="stub_model_id",
            model="stub",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=ARBITRARY_STRING,
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=10,
                total_tokens=20,
            ),
        )
        return response

    def check_model_config(self):
        r"""Directly pass the check on arguments to STUB model.
        """
        pass
