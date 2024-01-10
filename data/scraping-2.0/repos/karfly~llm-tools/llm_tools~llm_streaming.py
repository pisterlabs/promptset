from typing import (
    AsyncIterator,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from tenacity import wait_exponential
import asyncio
from tenacity.wait import wait_base
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import BaseMessage
from langchain.chat_models.openai import _convert_dict_to_message
import tiktoken
import openai
import openai.error

from concurrent.futures import Executor
from functools import partial

from llm_tools.chat_message import OpenAIChatMessage, prepare_messages
from llm_tools.tokens import (
    TokenExpense,
    TokenExpenses,
    count_tokens_from_input_messages,
    count_tokens_from_output_text,
)

from llm_tools.errors import (
    should_retry_initital_openai_request_error,
    should_retry_streaming_openai_request_error,
    should_fallback_to_other_model,
    get_openai_retrying_iterator,
    ModelContextSizeExceededError,
    StreamingNextTokenTimeoutError,
    OpenAIRequestTimeoutError,
    CONTEXT_LENGTH_EXCEEDED_ERROR_CODE,
    MultipleException,
)
from llm_tools.llm_streaming_base import StreamingLLMBase



class StreamingOpenAIChatModel(StreamingLLMBase):
    def __init__(
        self,
        chat_model: Union[ChatOpenAI, AzureChatOpenAI],
        max_initial_request_retries: int = 5,
        max_streaming_retries: int = 2,
        wait_between_retries=wait_exponential(multiplier=1, min=1, max=60),
        streaming_next_token_timeout: int = 10,
        request_timeout: wait_base = wait_exponential(multiplier=1, min=5, max=60),
        token_count_executor: Optional[Executor] = None,
    ):
        self.chat_model = chat_model
        self.encoding = tiktoken.encoding_for_model(self.chat_model.model_name)
        self.max_request_retries = max_initial_request_retries
        self.max_streaming_retries = max_streaming_retries
        self.wait_between_retries = wait_between_retries
        self.streaming_next_token_timeout = streaming_next_token_timeout
        self.request_timeout = request_timeout
        self.token_count_executor = token_count_executor
        self.reset()

    @property
    def context_size(self) -> int:
        model_name = self.chat_model.model_name
        is_azure = isinstance(self.chat_model, AzureChatOpenAI)
        if is_azure:
            return {
                "gpt-3.5-turbo": 8192,
                "gpt-4": 8192,
            }[model_name]
        else:
            return {
                "gpt-3.5-turbo": 4097,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-4": 8192,
                "gpt-4-1106-preview": 128000,
            }[model_name]

    def reset(self):
        self.completions = []
        self.successful_request_attempts = 0
        self.request_attempts = 0
        self.streaming_attempts = 0
        self.message_dicts = None
        self._succeeded = False
        self.input_messages_n_tokens = 0
        self.output_tokens_spent_per_completion = []

    @property
    def succeeded(self) -> bool:
        return self._succeeded

    def prepare_messages(self, messages: List[OpenAIChatMessage]) -> List[BaseMessage]:
        result = []
        for message in messages:
            if not isinstance(message, BaseMessage):
                message = _convert_dict_to_message(message)
            result.append(message)
        return result

    async def stream_llm_reply(
        self,
        messages: List[OpenAIChatMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        assert self.chat_model.streaming
        assert len(messages) > 0
        self.reset()
        _f = partial(count_tokens_from_input_messages,
            messages=messages,
            model_name=self.chat_model.model_name,
        )
        if self.token_count_executor is None:
            self.input_messages_n_tokens = _f()
        else:
            self.input_messages_n_tokens = await asyncio.get_running_loop().run_in_executor(
                self.token_count_executor,
                _f,
            )
        if self.input_messages_n_tokens > self.context_size:
            raise ModelContextSizeExceededError(
                model_name=self.chat_model.model_name,
                max_context_length=self.context_size,
                context_length=self.input_messages_n_tokens,
                during_streaming=False,
            )

        self.message_dicts, params = self.chat_model._create_message_dicts(
            messages=prepare_messages(messages),
            stop=stop,
        )
        params["stream"] = True

        async for streaming_attempt in get_openai_retrying_iterator(
            retry_if_exception_fn=should_retry_streaming_openai_request_error,
            max_retries=self.max_streaming_retries,
            wait=self.wait_between_retries,
        ):
            completion = ""
            role = "assistant"
            self.streaming_attempts += 1
            self.output_tokens_spent_per_completion.append(0)

            async for request_attempt in get_openai_retrying_iterator(
                retry_if_exception_fn=should_retry_initital_openai_request_error,
                max_retries=self.max_request_retries,
                wait=self.wait_between_retries,
            ):
                with request_attempt:
                    self.request_attempts += 1
                    timeout = self.request_timeout(request_attempt.retry_state)
                    try:
                        gen = await asyncio.wait_for(
                            self.chat_model.client.acreate(messages=self.message_dicts, **params),
                            timeout=timeout,
                        )
                    except openai.error.InvalidRequestError as e:
                        if e.code == CONTEXT_LENGTH_EXCEEDED_ERROR_CODE:
                            raise ModelContextSizeExceededError.from_openai_error(
                                model_name=self.chat_model.model_name,
                                during_streaming=False,
                                error=e,
                            ) from e
                        else:
                            raise
                    except asyncio.TimeoutError as e:
                        raise OpenAIRequestTimeoutError() from e
                    except:
                        raise
                    else:
                        self.successful_request_attempts += 1

            with streaming_attempt:
                try:
                    gen_iter = gen.__aiter__()
                    while True:
                        try:
                            stream_resp = await asyncio.wait_for(
                                gen_iter.__anext__(),
                                timeout=self.streaming_next_token_timeout,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as e:
                            raise StreamingNextTokenTimeoutError() from e
                        finish_reason = stream_resp["choices"][0].get("finish_reason")
                        role = stream_resp["choices"][0]["delta"].get("role", role)
                        token = stream_resp["choices"][0]["delta"].get("content", "")

                        _f = partial(count_tokens_from_output_text,
                            text=token,
                            model_name=self.chat_model.model_name,
                        )
                        if self.token_count_executor is None:
                            _tokens = _f()
                        else:
                            _tokens = await asyncio.get_running_loop().run_in_executor(
                                self.token_count_executor,
                                _f,
                            )
                        self.output_tokens_spent_per_completion[-1] += _tokens
                        completion += token
                        if token:
                            yield completion, token
                        if finish_reason:
                            if finish_reason == "length":
                                raise ModelContextSizeExceededError(
                                    model_name=self.chat_model.model_name,
                                    max_context_length=self.context_size,
                                    context_length=self.input_messages_n_tokens + self.output_tokens_spent_per_completion[-1],
                                    during_streaming=True,
                                )
                            elif finish_reason != "stop":
                                raise ValueError(f"Unknown finish reason: {finish_reason}")
                finally:
                    self.completions.append(completion)

        self._succeeded = True

    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> TokenExpenses:
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")

        n_input_tokens_per_trial = self.input_messages_n_tokens
        if only_successful_trial:
            n_input_tokens = n_input_tokens_per_trial
            n_output_tokens = self.output_tokens_spent_per_completion[-1]
        else:
            n_input_tokens = n_input_tokens_per_trial * self.successful_request_attempts
            n_output_tokens = sum(self.output_tokens_spent_per_completion)
        expenses = TokenExpenses()
        expense = TokenExpense(
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens,
            model_name=self.chat_model.model_name,
        )
        expenses.add_expense(expense)
        return expenses
