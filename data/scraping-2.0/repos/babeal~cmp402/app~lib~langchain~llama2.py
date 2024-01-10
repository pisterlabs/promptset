import json
import warnings
from abc import ABC
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LineIterator
from langchain.llms.bedrock import BedrockBase
from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.schema.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain.schema.output import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    GenerationChunk,
)
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)

# Special tokens
BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class SageMakerContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        return input_str

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"][0]


class Llama2Chat(BaseChatModel, SagemakerEndpoint):
    """A chat model for Llama 2 on SageMaker."""

    content_handler: LLMContentHandler = SageMakerContentHandler()

    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 0.9

    max_tokens_to_sample = 200

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llama2_chat_sm"

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "temperature": self.temperature,
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "top_p": self.top_p,
        }

    def _return_role_name(self, message):
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            raise ValueError(f"Got unknown type {message}")

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        messages = [
            {"role": self._return_role_name(message), "content": message.content}
            for message in messages
        ]

        if messages[0]["role"] == "system":
            messages = [
                {
                    "role": messages[1]["role"],
                    "content": B_SYS
                    + messages[0]["content"]
                    + E_SYS
                    + messages[1]["content"],
                }
            ] + messages[2:]

        assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )

        prompts = [
            f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}{EOS}"
            for prompt, answer in zip(
                messages[::2],
                messages[1::2],
            )
        ]

        assert (
            messages[-1]["role"] == "user"
        ), f"Last message must be from user, got {messages[-1]['role']}"
        human_message_prompt = (
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}"
        )

        prompts.append(human_message_prompt)

        return "\n".join(prompts)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _model_kwargs = {
            **self._default_params,
            **self.model_kwargs,
            **kwargs,
        }

        prompt = self._format_messages_as_text(messages)

        text = super()._call(
            prompt,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **_model_kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=text.strip()),
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        _model_kwargs = {
            **self._default_params,
            **self.model_kwargs,
            **kwargs,
        }
        _endpoint_kwargs = self.endpoint_kwargs or {}

        prompt = self._format_messages_as_text(messages)

        body = self.content_handler.transform_input(prompt, _model_kwargs)

        try:
            resp = self.client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=self.content_handler.content_type,
                **_endpoint_kwargs,
            )
            iterator = LineIterator(resp["Body"])
            current_completion: str = ""
            for line in iterator:
                resp = json.loads(line)
                resp_output = resp.get("outputs")[0]
                if stop is not None:
                    # Uses same approach as below
                    resp_output = enforce_stop_tokens(resp_output, stop)
                current_completion += resp_output
                if resp_output:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=resp_output)
                    )
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(resp_output)
        except Exception as e:
            raise ValueError(f"Error raised by streaming inference endpoint: {e}")
