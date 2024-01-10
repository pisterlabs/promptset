"""
Wrapper to choose between a few OpenAI keys before chat generation
"""
import copy
from typing import Any, AsyncIterator, Iterator, List, Union
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.output import ChatGenerationChunk, ChatResult
from langchain.schema.messages import BaseMessage
from .capture_headers import attach_session_hooks
from .limit_info import choose_key, achoose_key, ApiKey
from .limit_await_chat_openai import LimitAwaitChatOpenAI


_LIMIT_AWAIT_SLEEP = 0.01
_LIMIT_AWAIT_TIMEOUT = 60.0


class ChooseKeyChatOpenAI(BaseChatModel):
    """
    Key-choosing OpenAI chat wrapper
    """
    chat_openai: BaseChatModel # Since pydantic do not allow polymorphism -
                               # we will use base model her
                               # than introduce specific property and validatior
    openai_api_keys: List[ApiKey] # API keys

    @property
    def _chat_model(self) -> Union[ChatOpenAI, LimitAwaitChatOpenAI]:
        """
        Get chat model
        """
        value = self.chat_openai
        assert isinstance(value, (ChatOpenAI, LimitAwaitChatOpenAI))
        return value

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        # pylint: disable=protected-access
        return self._chat_model._llm_type
        # pylint: enable=protected-access

    @property
    def model_name(self) -> str:
        return self._chat_model.model_name

    def get_num_tokens(self, text: str) -> int:
        """
        Calculates number of tokens
        """
        return self._chat_model.get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Calculates number of tokens
        """
        return self._chat_model.get_num_tokens_from_messages(messages)

    def _stream(self, messages: List[BaseMessage],
                stop: List[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        token_count = self.get_num_tokens_from_messages(messages)
        chat_openai = copy.deepcopy(self._chat_model)
        chat_openai.openai_api_key = choose_key(chat_openai.model_name,
                                                self.openai_api_keys,
                                                token_count)
        # pylint: disable=protected-access
        for chunk in chat_openai._stream(messages, stop, run_manager, **kwargs):
            yield chunk
        # pylint: enable=protected-access

    # pylint: disable=invalid-overridden-method
    # I need to perform async operations inside, so method is async - and it works this way
    async def _astream(self, messages: List[BaseMessage],
                       stop: List[str] | None = None,
                       run_manager: AsyncCallbackManagerForLLMRun | None = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        token_count = self.get_num_tokens_from_messages(messages)
        chat_openai = copy.deepcopy(self._chat_model)
        chat_openai.openai_api_key = await achoose_key(chat_openai.model_name,
                                                       self.openai_api_keys,
                                                       token_count)
        # pylint: disable=protected-access
        async for chunk in chat_openai._astream(messages, stop, run_manager, **kwargs):
            yield chunk
        # pylint: enable=protected-access
    # pylint: enable=invalid-overridden-method

    def _generate(self, messages: List[BaseMessage],
                  stop: List[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        token_count = self.get_num_tokens_from_messages(messages)
        chat_openai = copy.deepcopy(self._chat_model)
        chosen_key = choose_key(chat_openai.model_name,
                                self.openai_api_keys,
                                token_count)
        chat_openai.openai_api_key = chosen_key
        # pylint: disable=protected-access
        return chat_openai._generate(messages,
                                     stop,
                                     run_manager,
                                     **kwargs)
        # pylint: enable=protected-access

    async def _agenerate(self, messages: List[BaseMessage],
                         stop: List[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None,
                         **kwargs: Any) -> ChatResult:
        token_count = self.get_num_tokens_from_messages(messages)
        chat_openai = copy.deepcopy(self._chat_model)
        chat_openai.openai_api_key = await achoose_key(chat_openai.model_name,
                                                       self.openai_api_keys,
                                                       token_count)
        # pylint: disable=protected-access
        return await chat_openai._agenerate(messages,
                                            stop,
                                            run_manager,
                                            **kwargs)
        # pylint: enable=protected-access


attach_session_hooks()
