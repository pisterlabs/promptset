# pylint: disable=all

from typing import Any, Coroutine, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult

class ChatOpenAI(BaseChatModel):
    def __init__(self, *, temperature: float = 0.7) -> None: ...
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = ...,
        run_manager: CallbackManagerForLLMRun | None = ...,
        **kwargs: Any
    ) -> ChatResult: ...
    def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = ...,
        run_manager: AsyncCallbackManagerForLLMRun | None = ...,
        **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResult]: ...
    def _llm_type(self) -> str: ...
