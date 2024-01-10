from typing import Any, Dict, List

from .base import BaseChat, BaseLlmModel, LLmInputInterface

from langchain.llms.fireworks import Fireworks
from langchain.chat_models.fireworks import ChatFireworks
from langchain.schema.output import LLMResult
from langchain.callbacks.base import Callbacks
from langchain.schema import BaseMessage


class FireworksModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = Fireworks(
            fireworks_api_key=input.api_key,
            model=input.model_name if input.model_name else "accounts/cresta-ai/models/openorca-7b-fast",
            max_retries=input.max_retries,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
            model_kwargs={"temperature": input.temperature,
                          "max_tokens": input.max_tokens, "top_p": input.top_p}
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            prompts=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result


class ChatFireworksModel(BaseChat):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = ChatFireworks(
            fireworks_api_key=input.api_key,
            model=input.model_name if input.model_name else "accounts/fireworks/models/llama-v2-13b-code-instruct",
            max_retries=input.max_retries,
            cache=input.cache,
            verbose=input.verbose,
            callbacks=input.callbacks,
            metadata=input.metadata,
            model_kwargs={"temperature": input.temperature,
                          "max_tokens": input.max_tokens, "top_p": input.top_p}
        )  # type: ignore

    def compelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            messages=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = await self.client.agenerate(messages=prompts, callbacks=callbacks, metadata=metadata)
        return result
