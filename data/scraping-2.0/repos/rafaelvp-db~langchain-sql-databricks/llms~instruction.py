"""This module contains a custom LLM class ready for usage in Langchain."""

from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from models.instruction import InstructionTextGenerationPipeline

class InstructionLLM(LLM):

  _pipeline = InstructionTextGenerationPipeline(model_name="mosaicml/mpt-7b-instruct")
        
  @property
  def _llm_type(self) -> str:
      return "custom"
  
  @classmethod
  def generate(
    self,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    callbacks = None
  ) -> str:
    
    response = self._pipeline.process_stream(
      instruction = prompt,
      temperature = temperature,
      top_p = top_p,
      top_k = top_k,
      max_new_tokens = max_new_tokens
    )

    return response
  
  def __call__(
    self,
    prompt,
    temperature = 0.3,
    top_p = 0.95,
    top_k = 0,
    max_new_tokens = 1000
  ):
    return self.generate(
      prompt=prompt,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      max_new_tokens=max_new_tokens
    )
  
  def _call(
    self,
    **kwargs
  ):
    pass