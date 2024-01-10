from typing import Annotated, Callable, Optional, Dict, Literal, Union, TypedDict
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chat_models.openai import ChatOpenAI

LLMSpec = Annotated[Union[
  "OpenAILLMSpec",
  "HuggingFaceHubLLMSpec",
  "ChatOpenAILLMSpec",
  ], Field(discriminator='llm_type')]


class BaseLLMSpec(BaseModel):
  def to_llm(self) -> LLM:
    raise NotImplementedError

  def copy_replace(self, replace: Callable[[LLMSpec], LLMSpec]):
    return replace(self).copy(deep=True)


class OpenAILLMSpec(BaseLLMSpec):
  llm_type: Literal["openai"] = "openai"
  model_name: str
  temperature: float
  max_tokens: int
  top_p: float
  frequency_penalty: float
  presence_penalty: float
  n: int
  request_timeout: Optional[int]
  logit_bias: Optional[Dict[int, int]]

  def to_llm(self) -> LLM:
    return OpenAI(model_name=self.model_name, temperature=self.temperature,
                  max_tokens=self.max_tokens, top_p=self.top_p, frequency_penalty=self.frequency_penalty,
                  presence_penalty=self.presence_penalty, n=self.n,
                  request_timeout=self.request_timeout, logit_bias=self.logit_bias)


class HuggingFaceHubLLMSpec(BaseLLMSpec):
  class ModelKwargs(TypedDict):
    temperature: float
    max_length: int

  llm_type: Literal["huggingface_hub"] = "huggingface_hub"
  repo_id: str
  task: Optional[str]
  model_kwargs: Optional[ModelKwargs]

  def to_llm(self) -> LLM:
    return HuggingFaceHub(model_kwargs=self.model_kwargs, repo_id=self.repo_id, task=self.task)


class ChatOpenAILLMSpec(BaseLLMSpec):
  llm_type: Literal["chat_openai"] = "chat_openai"
  model_name: str
  temperature: float
  max_tokens: int
  n: int
  request_timeout: Optional[int]

  def to_llm(self) -> LLM:
    return ChatOpenAI(model_name=self.model_name, temperature=self.temperature,
                      max_tokens=self.max_tokens, n=self.n, request_timeout=self.request_timeout)
