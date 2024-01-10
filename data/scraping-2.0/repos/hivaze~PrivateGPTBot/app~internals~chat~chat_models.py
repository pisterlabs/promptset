import json
import logging
import time
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
import tiktoken
from openai.openai_object import OpenAIObject

from app import settings
from app.internals.chat.chat_history import ChatHistory, ChatMessage, ChatRole, FunctionCallMessage, \
    FunctionResponseMessage
from app.settings import ModelConfig
from app.utils.misc import percent_trim_list

logger = logging.getLogger(__name__)

MAX_HIST_LEN = settings.config.last_messages_count
openai.api_key = settings.config.OPENAI_KEY


@dataclass
class TextGenerationResult:
    message: ChatMessage
    is_function_call: bool
    time_taken: int  # in ms
    prompt_tokens_usage: int
    completion_tokens_usage: int
    total_tokens_usage: int
    retires_count: int
    model_config: ModelConfig


class BaseChatModel(ABC):

    def __init__(self, config: ModelConfig):
        self.config = config
        tokens_field = 'max_tokens' if 'max_tokens' in self.config.generation_params.keys() else 'max_new_tokens'
        self.max_gen_tokens = self.config.generation_params.get(tokens_field, 856)

    @abstractmethod
    def tokenize_sentence(self, message: str) -> list:
        pass

    @abstractmethod
    def detokenize_sentence(self, tokens: list) -> str:
        pass

    def _count_str_tokens(self, text: str) -> int:
        return len(self.tokenize_sentence(text))

    @abstractmethod
    def count_tokens(self, message: ChatMessage) -> int:
        pass

    @abstractmethod
    def count_functions_prompt_tokens(self, functions: list) -> int:
        pass

    @abstractmethod
    def count_tokens_overflow(self, history: ChatHistory, functions: list) -> typing.Tuple[int, list]:
        pass

    def _truncate_history(self, history: ChatHistory, functions: list):
        """
        Truncates chat history in three ways:
        1) Removes messages from the beginning those go beyond the allowed history length
        2) If history has many message: Drops messages from the beginning of the history until needed amount of free tokens reached
        3) If history has only one message - iteratively trim that message from the end by 10% of its length
        """
        tokens_to_remove, chat_history = self.count_tokens_overflow(history, functions)

        while tokens_to_remove > 0:
            if len(chat_history) > 1 and type(chat_history[-1]) is not FunctionResponseMessage:
                dropped_message = chat_history.pop(0)
                tokens_to_remove -= self.count_tokens(dropped_message)
            else:
                tokens = self.tokenize_sentence(chat_history[-1].text)
                print(tokens_to_remove, tokens_to_remove / len(tokens))
                new_tokens = percent_trim_list(tokens, percent=min(tokens_to_remove / len(tokens), 0.05))
                tokens_to_remove -= len(tokens) - len(new_tokens)
                # tokens_to_remove -= (len(tokens) - len(new_tokens))
                chat_history[-1].text = self.detokenize_sentence(new_tokens)

        history.chat_history = chat_history

    def generate_answer(self, history: ChatHistory, functions=None, function_call=None) -> TextGenerationResult:
        self._truncate_history(history, functions)

        formatted_history = self._format_history(history)
        return self._generate_answer(formatted_history, functions, function_call)

    @abstractmethod
    def _generate_answer(self, formatted_history, functions, function_call) -> TextGenerationResult:
        pass

    @abstractmethod
    def _is_function_call(self, message_container) -> bool:
        pass

    @abstractmethod
    def _format_message(self, message: ChatMessage) -> object:
        pass

    @abstractmethod
    def _format_history(self, history: ChatHistory) -> object:
        pass

    @abstractmethod
    def _parse_output(self, message_container, is_function_call: bool) -> ChatMessage:
        pass


class OpenAIChatModel(BaseChatModel):
    ROLES_TEXT_MAPPING = {
        ChatRole.SYSTEM: 'system',
        ChatRole.USER: 'user',
        ChatRole.ASSISTANT: 'assistant',
        ChatRole.FUNCTION: 'function'
    }
    TEXT_ROLES_MAPPING = {v: k for k, v in ROLES_TEXT_MAPPING.items()}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def tokenize_sentence(self, message: str) -> list:
        return self.tokenizer.encode(message)

    def detokenize_sentence(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)

    def count_tokens(self, message: ChatMessage) -> int:
        if type(message) == ChatMessage:
            return self._count_str_tokens(message.text)
        elif type(message) == FunctionCallMessage:
            arguments_tokens = 0
            for k, v in message.arguments.items():
                arguments_tokens += self._count_str_tokens(k)
                arguments_tokens += self._count_str_tokens(v)
            return self._count_str_tokens(message.name) + arguments_tokens
        elif type(message) == FunctionResponseMessage:
            return self._count_str_tokens(message.name) + self._count_str_tokens(message.text)

    def count_functions_prompt_tokens(self, functions: list) -> int:
        functions_tokens = 0
        for function in functions:
            functions_tokens += self._count_str_tokens(function['name'])
            functions_tokens += self._count_str_tokens(function['description'])
            for k, v in function['parameters']['properties'].items():
                functions_tokens += self._count_str_tokens(k)
                functions_tokens += self._count_str_tokens(v['description'])
        return functions_tokens

    def count_tokens_overflow(self, history: ChatHistory, functions: list) -> typing.Tuple[int, list]:
        chat_history = history.chat_history[-MAX_HIST_LEN:]

        total_tokens = sum([self.count_tokens(message) for message in chat_history])
        total_tokens += 4 * len(chat_history)
        if history.system_message:
            total_tokens += self.count_tokens(history.system_message) + 11
        if functions:
            total_tokens += self.count_functions_prompt_tokens(functions) + 50

        tokens_to_remove = max(total_tokens + self.max_gen_tokens - self.config.max_context_size, 0)
        return tokens_to_remove, chat_history

    def _format_message(self, message: ChatMessage) -> object:
        if isinstance(message, FunctionResponseMessage):
            return {"role": self.ROLES_TEXT_MAPPING[ChatRole.FUNCTION],
                    "content": message.text,
                    "name": message.name}
        elif isinstance(message, FunctionCallMessage):
            return {"role": self.ROLES_TEXT_MAPPING[ChatRole.ASSISTANT],
                    "content": None,
                    "function_call": {"name": message.name, "arguments": json.dumps(message.arguments,
                                                                                    ensure_ascii=False)}}
        else:
            return {"role": self.ROLES_TEXT_MAPPING[message.role],
                    "content": message.text}

    def _format_history(self, history: ChatHistory):
        if history.system_message is not None:
            openai_hist = [self._format_message(history.system_message)]
        else:
            openai_hist = []
        return openai_hist + [
            self._format_message(message)
            for message in history.chat_history
        ]

    def _is_function_call(self, message_container) -> bool:
        return message_container.get("function_call") is not None

    def _parse_output(self, message_container, is_function_call: bool) -> ChatMessage:
        if is_function_call:
            return FunctionCallMessage(name=message_container['function_call']['name'],
                                       arguments=json.loads(message_container['function_call']['arguments']))
        return ChatMessage(role=self.TEXT_ROLES_MAPPING[message_container['role']], text=message_container['content'])

    def _generate_answer(self, formatted_history, functions, function_call) -> TextGenerationResult:
        start_time = int(time.time() * 1000)
        for i in range(settings.config.openai_api_retries):
            try:
                if functions is not None and len(functions) > 0:  # openai.error.InvalidRequestError fix
                    response: OpenAIObject = openai.ChatCompletion.create(messages=formatted_history,
                                                                          model=self.config.model_name,
                                                                          functions=functions,
                                                                          function_call=function_call,
                                                                          **self.config.generation_params)
                else:
                    response: OpenAIObject = openai.ChatCompletion.create(messages=formatted_history,
                                                                          model=self.config.model_name,
                                                                          **self.config.generation_params)
                time_taken = int(time.time() * 1000) - start_time
                is_function_call = self._is_function_call(response['choices'][0]['message'])
                chat_message = self._parse_output(response['choices'][0]['message'], is_function_call)
                return TextGenerationResult(message=chat_message,
                                            time_taken=time_taken,
                                            is_function_call=is_function_call,
                                            prompt_tokens_usage=response['usage']['prompt_tokens'],
                                            completion_tokens_usage=response['usage']['completion_tokens'],
                                            total_tokens_usage=response['usage']['total_tokens'],
                                            model_config=self.config,
                                            retires_count=i)
            except (openai.error.APIError, openai.error.RateLimitError) as e:
                logger.warning(f"Got exception from OpenAI: {e}")
                time.sleep(2 ** i)  # wait longer


def load_chat_model(model_config: ModelConfig) -> BaseChatModel:
    assert model_config.type in ['open-ai'], f"{model_config.type} is not supported model type"
    if model_config.type == 'open-ai':
        return OpenAIChatModel(model_config)
