import json

from typing import Any, Mapping
import requests
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from requests.models import Response

FASTCHAT_URL = "TODO"
DEFAULT_TEMPRATURE = 0.2

RESPONSE_CHUNK_DELIMIETER = b"\0"
TEXT_FIELD_NAME_IN_RESPONSE_CHUNK = "text"
TEXT_RESPONSE_DEFAULT_VALUE = ''
FASTCHAT_LLM_NAME = "fastchat_llm"
LLM_DEFAULT_IDENTIFYING_PARAMETERS = {}


class FastchatLLM(LLM):
    _fastchat_model: str = PrivateAttr(default_factory=str)
    _temprature: str = PrivateAttr(default_factory=str)

    def __init__(self, fastchat_model: str, temperature: float = DEFAULT_TEMPRATURE):
        super().__init__()
        self._fastchat_model = fastchat_model
        self._temperature = temperature


    @property
    def _llm_type(self) -> str:
        return FASTCHAT_LLM_NAME

    @property
    def _identyfing_params(self) -> Mapping[str, Any]:
        return LLM_DEFAULT_IDENTIFYING_PARAMETERS

    def _call(self, prompt: str, **kwargs: Any) -> str:
        model_request_payload = self._build_model_request_payload(prompt)

        response = requests.post(FASTCHAT_URL, json=model_request_payload.dict())
        response.raise_for_status()

        content = _get_model_response_from_stream(response)

        return content

    def _build_model_request_payload(self, prompt: str) -> FastChatRequestPayload:
        payload = FastChatRequestPayload(model=self._fastchat_model, prompt=prompt, temprature=self._temprature)
        return payload


def _get_model_response_from_stream(response: Response) -> str:
    text_response = TEXT_RESPONSE_DEFAULT_VALUE
    for chunk in response.iter_lines(decode_unicode=False, delimiter=RESPONSE_CHUNK_DELIMIETER):
        if chunk:
            data = json.loads(chunk.decode())
            chunk_text_response = data[TEXT_FIELD_NAME_IN_RESPONSE_CHUNK]
            text_response = chunk_text_response

    return text_response
