from itertools import tee
import json
import logging
import os
import uuid

from typing import Any, Dict, Iterable, Tuple, cast, overload

from openai._types import NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import (
    Function,
    ChatCompletionMessageToolCall,
)
from openai.types.completion import Completion

from .pii import Redactor
from .session import event_session
from .template import TemplateChat, TemplateString
from .types import LibrettoConfig, LibrettoCreateParamDict, LibrettoCreateParams


logger = logging.getLogger(__name__)


class LibrettoCompletionsBaseMixin:
    def __init__(self, config: LibrettoConfig) -> None:
        self.config = config
        self.pii_redactor = Redactor() if config.redact_pii else None

    def _original_create(self, **kwargs):
        raise NotImplementedError()

    def _create(self, *, libretto: LibrettoCreateParamDict | None, **original_kwargs):
        libretto = self._prepare_create_params(libretto, **original_kwargs)

        if libretto["project_key"] is None and libretto["api_key"] is None:
            return self._original_create(**original_kwargs)

        if libretto["prompt_template_name"] is None and not self.config.allow_unnamed_prompts:
            return self._original_create(**original_kwargs)

        model_params = self._build_model_params(**original_kwargs)
        tools = model_params.pop("tools", None)

        # Redact PII from template parameters if configured to do so
        self._redact_template_params(libretto["template_params"])

        with event_session(
            project_key=libretto["project_key"],
            api_key=libretto["api_key"],
            prompt_template_name=libretto["prompt_template_name"],
            model_params=model_params,
            prompt_template_text=libretto["template_text"],
            prompt_template_chat=libretto["template_chat"],
            chat_id=libretto["chat_id"],
            prompt_template_params=libretto["template_params"],
            prompt_event_id=libretto["event_id"],
            parent_event_id=libretto["parent_event_id"],
            feedback_key=libretto["feedback_key"],
            tools=tools,
        ) as complete_event:
            response = self._original_create(**original_kwargs)
            return_response, event_response = self._get_result(response)

            # Can only do this for non-streamed responses right now
            if hasattr(return_response, "model_copy"):
                return_response = return_response.model_copy(
                    update={"libretto_feedback_key": libretto["feedback_key"]}
                )

            # Redact PII before recording the event
            event_response = self._redact_response(event_response)

            complete_event(event_response)

            return return_response

    def _prepare_create_params(
        self,
        libretto: LibrettoCreateParamDict | None = None,
        **_original_kwargs,
    ) -> LibrettoCreateParamDict:
        if libretto is None:
            libretto = LibrettoCreateParams()
        else:
            # Don't mutate the input dict
            libretto = libretto.copy()

        libretto["prompt_template_name"] = (
            libretto["prompt_template_name"]
            or self.config.prompt_template_name
            or os.environ.get("LIBRETTO_TEMPLATE_NAME")
            or libretto["api_name"]  # legacy
        )
        libretto["api_key"] = (
            libretto["api_key"] or self.config.api_key or os.environ.get("LIBRETTO_API_KEY")
        )
        libretto["chat_id"] = (
            libretto["chat_id"] or self.config.chat_id or os.environ.get("LIBRETTO_CHAT_ID")
        )
        libretto["project_key"] = libretto["project_key"] or os.environ.get("LIBRETTO_PROJECT_KEY")
        libretto["feedback_key"] = libretto["feedback_key"] or str(uuid.uuid4())

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = {"modelProvider": "openai"}
        for k, v in original_kwargs.items():
            if isinstance(v, NotGiven) or v is None:
                continue
            model_params[k] = v
        return model_params

    def _redact_template_params(self, template_params) -> None:
        if not template_params or not self.pii_redactor:
            return

        for name, param in template_params.items():
            try:
                template_params[name] = self.pii_redactor.redact(param)
            except Exception as e:
                logger.warning(
                    "Failed to redact PII from parameter: key=%s, value=%s, error=%s",
                    name,
                    param,
                    e,
                )

    def _redact_response(self, event_response: str | None) -> str:
        if not event_response:
            return ""
        if not self.pii_redactor:
            return event_response

        try:
            return self.pii_redactor.redact_text(event_response)
        except Exception as e:
            logger.warning(
                "Failed to redact PII from response: error=%s",
                e,
            )
            return event_response

    @overload
    def _get_result(
        self, response: ChatCompletion | Iterable[ChatCompletionChunk]
    ) -> Tuple[ChatCompletion | Iterable[ChatCompletionChunk], str]:
        ...

    @overload
    def _get_result(
        self, response: Completion | Iterable[Completion]
    ) -> Tuple[Completion | Iterable[Completion], str]:
        ...

    def _get_result(
        self,
        response: Completion
        | Iterable[Completion]
        | ChatCompletion
        | Iterable[ChatCompletionChunk],
    ) -> Tuple[
        Completion | Iterable[Completion] | ChatCompletion | Iterable[ChatCompletionChunk], str
    ]:
        raise NotImplementedError()


class LibrettoCompletionsMixin(LibrettoCompletionsBaseMixin):
    def _original_create(self, *args, **kwargs):
        raise NotImplementedError()

    def _prepare_create_params(
        self,
        libretto: LibrettoCreateParamDict | None = None,
        **original_kwargs,
    ) -> LibrettoCreateParamDict:
        libretto = super()._prepare_create_params(libretto=libretto, **original_kwargs)

        prompt = original_kwargs.get("prompt", "")
        if isinstance(prompt, TemplateString):
            libretto["template_text"] = prompt.template
            if libretto["template_params"] is None:
                libretto["template_params"] = {}
            libretto["template_params"].update(prompt.params)
        else:
            libretto["template_text"] = prompt

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = super()._build_model_params(**original_kwargs)
        model_params["modelType"] = "completion"
        del model_params["prompt"]
        return model_params

    def _get_result(
        self, response: Completion | Iterable[Completion]
    ) -> Tuple[Completion | Iterable[Completion], str]:
        if isinstance(response, Completion):
            return response, response.choices[0].text

        original, consumable = tee(response)
        accumulated = []
        for response_chunk in consumable:
            accumulated.append(response_chunk.choices[0].text)
        return (original, "".join(accumulated))


class LibrettoChatCompletionsMixin(LibrettoCompletionsBaseMixin):
    def _original_create(self, *args, **kwargs):
        raise NotImplementedError()

    def _prepare_create_params(
        self,
        libretto: LibrettoCreateParamDict | None = None,
        **original_kwargs,
    ) -> LibrettoCreateParamDict:
        libretto = super()._prepare_create_params(libretto=libretto, **original_kwargs)

        messages = original_kwargs.get("messages")
        if hasattr(messages, "template"):
            libretto["template_chat"] = cast(TemplateChat, messages).template
        if hasattr(messages, "params"):
            if libretto["template_params"] is None:
                libretto["template_params"] = {}
            libretto["template_params"].update(cast(TemplateChat, messages).params)

        return libretto

    def _build_model_params(self, **original_kwargs) -> Dict[str, Any]:
        model_params = super()._build_model_params(**original_kwargs)
        model_params["modelType"] = "chat"
        del model_params["messages"]
        return model_params

    def _get_result(
        self, response: ChatCompletion | Iterable[ChatCompletionChunk]
    ) -> Tuple[ChatCompletion | Iterable[ChatCompletionChunk], str]:
        if not isinstance(response, ChatCompletion):
            return self._get_stream_result(response)

        message = response.choices[0].message
        if message.function_call:
            return response, json.dumps({"function_call": message.function_call.model_dump()})
        if message.tool_calls:
            return response, json.dumps(
                {"tool_calls": [c.model_dump() for c in message.tool_calls]}
            )
        return response, message.content or ""

    def _get_stream_result(
        self, responses: Iterable[ChatCompletionChunk]
    ) -> Tuple[Iterable[ChatCompletionChunk], str]:
        (original_response, consumable_response) = tee(responses)
        accumulated = []
        for response in consumable_response:
            if response.choices[0].delta.content is not None:
                accumulated.append(response.choices[0].delta.content)
            if response.choices[0].delta.function_call is not None:
                logger.warning("Streaming a function_call response is not supported yet.")
        return (original_response, "".join(accumulated))
