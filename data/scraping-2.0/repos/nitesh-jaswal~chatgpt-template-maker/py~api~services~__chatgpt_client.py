from __future__ import annotations

import logging 
from api.settings import OpenAIAuthSettings, OpenAIAPISettings
from api.exceptions import OpenAIClientAuthException, OpenAIAPIException , OpenAIErrorKind
from api.models.openai_models import Role, Prompt, PromptBuffer
from api.models.openai_models import OpenAIChatResponse, OpenAIChatRequest, FinishReason
from typing import Dict, List, cast
from .__baseclient import BaseClient

DEFAULT_SYSTEM_PROMPT = Prompt(
    role=Role.SYSTEM,
    content="Please answer in less than 200 words the response to the following query"
)

def _get_logger() -> logging.Logger:
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    return logger

class ChatGPTClient:
    auth: OpenAIAuthSettings | None = None

    def __init__(self, api_settings: OpenAIAPISettings, logger: logging.Logger | None = None):
        self.logger = logger if logger else _get_logger()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if api_settings.system_prompt is None else api_settings.system_prompt
        self._buffer: PromptBuffer = PromptBuffer(api_settings.max_prompts)
        self.api_settings = api_settings
        self._client = self.__create_client()

    def __create_client(self):
        if self.auth is None:
            raise OpenAIClientAuthException("Client auth settings not available. Please verify")
        return BaseClient(self.auth.api_key.get_secret_value(), self.auth.organization)

    def buffer_length(self) -> int:
        return len(self._buffer)
    # TODO: Change interface to List[str; Also tokenizer]
    def add_messages(self, messages: List[Prompt]) -> ChatGPTClient:
        self._buffer.extend(messages)
        return self
    
    def send_messages(self) -> Prompt:
        """`None` or `raises OpenAIAPIException(ErrorKind.Enum)`"""
        request =  OpenAIChatRequest(
            model="gpt-3.5-turbo",
            messages=self._buffer.to_list(),
            max_tokens=10,
            logit_bias={"50256": -100},
            user="asodioasijd"
        ) # type: ignore
        raw: OpenAIChatResponse = self._client.send(request)
        
        if len(raw.choices) < 1:
            raise OpenAIAPIException(OpenAIErrorKind.EMPTY, "No chat responses were received")
        
        last_choice = raw.choices[-1]
        
        if last_choice.message.role != Role.ASSISTANT:
            raise OpenAIAPIException(OpenAIErrorKind.INTERNAL_ERROR, f"Incorrect role in chat response. Expected `Assistant` role, received `{last_choice.message.role}`")

        match last_choice:
            case _: return last_choice.message

    @property
    def buffer_list(self) -> List[Prompt]:
        return cast(List[Prompt], self._buffer[:] )