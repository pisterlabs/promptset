"""
Wrapper for ChatOpenAI which do limit awaiting before running the model
"""
from typing import Any, AsyncIterator, Coroutine, Iterator, List
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGenerationChunk, ChatResult
from .capture_headers import attach_session_hooks
from .limit_info import wait_for_limit, await_for_limit


_LIMIT_AWAIT_SLEEP = 0.01
_LIMIT_AWAIT_TIMEOUT = 60.0


class LimitAwaitChatOpenAI(BaseChatModel):
    """
    Rate/Token Per Minute waiting ChatOpenAI wrapper
    """
    chat_openai: ChatOpenAI
    limit_await_timeout: float = _LIMIT_AWAIT_TIMEOUT
    limit_await_sleep: float = _LIMIT_AWAIT_SLEEP
    openai_api_key: str = ""

    @property
    def model_name(self) -> str:
        return self.chat_openai.model_name

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        # pylint: disable=protected-access
        return self.chat_openai._llm_type
        # pylint: enable=protected-access

    def __getattribute__(self, __name: str) -> Any:
        if __name == "openai_api_key":
            return self.chat_openai.openai_api_key
        return super().__getattribute__(__name)

    def __setattr__(self, name, value) -> None:
        if name == "openai_api_key":
            self.chat_openai.openai_api_key = value
        else:
            super().__setattr__(name, value)

    @property
    def model_name(self) -> str:
        """
        Return OpenAI model name
        """
        return self.chat_openai.model_name

    def get_num_tokens(self, text: str) -> int:
        """
        Calculates number of tokens
        """
        return self.chat_openai.get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Calculates number of tokens
        """
        return self.chat_openai.get_num_tokens_from_messages(messages)

    def _stream(self, messages: List[BaseMessage],
                stop: List[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        token_count = self.get_num_tokens_from_messages(messages)
        wait_for_limit(
            self.model_name,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        # pylint: disable=protected-access
        for chunk in self.chat_openai._stream(messages, stop, run_manager, **kwargs):
            yield chunk
        # pylint: enable=protected-access

    # pylint: disable=invalid-overridden-method
    # I need to perform async operations inside, so method is async - and it works this way
    async def _astream(self, messages: List[BaseMessage],
                       stop: List[str] | None = None,
                       run_manager: AsyncCallbackManagerForLLMRun | None = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        token_count = self.get_num_tokens_from_messages(messages)
        await await_for_limit(
            self.model_name,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        # pylint: disable=protected-access
        async for chunk in self.chat_openai._astream(messages, stop, run_manager, **kwargs):
            yield chunk
        # pylint: enable=protected-access
    # pylint: enable=invalid-overridden-method

    def _generate(self, messages: List[BaseMessage],
                  stop: List[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        token_count = self.get_num_tokens_from_messages(messages)
        wait_for_limit(
            self.model_name,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        # pylint: disable=protected-access
        return self.chat_openai._generate(messages, stop, run_manager, **kwargs)
        # pylint: enable=protected-access

    async def _agenerate(self, messages: List[BaseMessage],
                         stop: List[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None,
                         **kwargs: Any) -> Coroutine[Any, Any, ChatResult]:
        token_count = self.get_num_tokens_from_messages(messages)
        await await_for_limit(
            self.model_name,
            self.openai_api_key,
            token_count,
            self.limit_await_timeout,
            self.limit_await_sleep,
        )
        # pylint: disable=protected-access
        return await self.chat_openai._agenerate(messages, stop, run_manager, **kwargs)
        # pylint: enable=protected-access


attach_session_hooks()
