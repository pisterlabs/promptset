import hashlib
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import requests
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_pydantic_field_names,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel, _generate_from_stream
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.sensenova-ai.com/v1"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
        _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


# signature generation
def _signature(secret_key: SecretStr, payload: Dict[str, Any], timestamp: int) -> str:
    input_str = secret_key.get_secret_value() + json.dumps(payload) + str(timestamp)
    md5 = hashlib.md5()
    md5.update(input_str.encode("utf-8"))
    return md5.hexdigest()


class Chatsensenova(BaseChatModel):
    """sensenova chat models API by sensenova Intelligent Technology.

    For more information, see https://platform.sensenova-ai.com/docs/api
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "sensenova_api_key": "sensenova_API_KEY",
            "sensenova_secret_key": "sensenova_SECRET_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    sensenova_api_base: str = Field(default=DEFAULT_API_BASE)
    """sensenova custom endpoints"""
    sensenova_api_key: Optional[str] = None
    """sensenova API Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = 60
    """request timeout for chat http requests"""

    primary_bot_name = "范闲"
    """primary bot name"""
    user_name = "林浦"
    """user name"""
    model = "082301_test"
    """model name of sensenova, default is `082301_test`."""
    tokens_to_generate = 500
    """How many tokens to generate."""
    beam_width: int = 4
    """Beam width for beam search."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["sensenova_api_base"] = get_from_dict_or_env(
            values,
            "sensenova_api_base",
            "sensenova_API_BASE",
            DEFAULT_API_BASE,
        )
        values["sensenova_api_key"] = get_from_dict_or_env(
            values,
            "sensenova_api_key",
            "sensenova_API_KEY",
        )
        values["sensenova_secret_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "sensenova_secret_key",
                "sensenova_SECRET_KEY",
            )
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling sensenova API."""
        normal_params = {
            "model": self.model,
            "tokens_to_generate": self.tokens_to_generate,
            "beam_width": self.beam_width,
        }

        return {**normal_params, **self.model_kwargs}

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return _generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)

        response = res.json()

        if response.get("code") != 0:
            raise ValueError(f"Error from sensenova api response: {response}")

        return self._create_chat_result(response)


    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        if self.sensenova_secret_key is None:
            raise ValueError("sensenova secret key is not set.")

        parameters = {**self._default_params, **kwargs}

        model = parameters.pop("model")
        beam_width = parameters.pop("beam_width")
        bot_name = parameters.pop("primary_bot_name")
        user_name = parameters.pop("user_name")
        tokens_to_generate = parameters.pop("tokens_to_generate")
        headers = parameters.pop("headers", {})

        payload = {
            "beam_width": beam_width,
            "tokens_to_generate" : tokens_to_generate,
            "model": model,
            "role_meta" : {
                "primary_bot_name": bot_name,
                "user_name": user_name,
            },
            "messages": [_convert_message_to_dict(m) for m in messages],
            "parameters": parameters,
        }

        url = self.sensenova_api_base
        url = f"{url}/v1/nlp/roleplay/completions"

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"{self.sensenova_api_key}",
                **headers,
            },
            json=payload,
        )
        return res

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = res["text"]
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "sensenova-chat"
