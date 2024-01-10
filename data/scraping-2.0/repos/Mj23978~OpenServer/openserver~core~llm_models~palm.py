from typing import Any, Dict, List, cast

from .base import BaseLlmModel, LLmInputInterface

from langchain.llms.google_palm import GooglePalm
from langchain.chat_models.google_palm import ChatGooglePalm
from langchain.schema.output import LLMResult
from langchain.callbacks.base import Callbacks, BaseCallbackManager
from langchain.schema import BaseMessage


class GooglePalmModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = GooglePalm(
            google_api_key=input.api_key,
            model_name=input.model_name if input.model_name else "models/text-bison-001",
            top_p=input.top_p,
            top_k=input.top_k,
            temperature=input.temperature,
            # max_output_tokens=input.max_tokens,            
            cache=input.cache,            
            verbose=input.verbose,
            callbacks=cast(BaseCallbackManager, input.callbacks),
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(prompts=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result


class ChatGooglePalmModel(BaseLlmModel):
    def __init__(self, input: LLmInputInterface) -> None:
        self.client = ChatGooglePalm(
            google_api_key=input.api_key,
            model_name=input.model_name if input.model_name else "models/chat-bison-001",
            top_p=input.top_p,
            top_k=input.top_k,
            temperature=input.temperature,
            # max_output_tokens=input.max_tokens,            
            cache=input.cache,            
            verbose=input.verbose,
            callbacks=cast(BaseCallbackManager, input.callbacks),
            metadata=input.metadata,
        )  # type: ignore

    def compelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(
            messages=prompts, callbacks=callbacks, metadata=metadata)
        return result

    async def acompelete(self, prompts: List[List[BaseMessage]], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = await self.client.agenerate(messages=prompts, callbacks=callbacks, metadata=metadata)
        return result
