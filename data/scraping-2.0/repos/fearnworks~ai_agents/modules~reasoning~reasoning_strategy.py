from langchain.llms import OpenAI
from pydantic import BaseModel
from langchain.llms.base import BaseLLM
from typing import Type, Callable



class ReasoningConfig(BaseModel):
    """
    A configuration class for the reasoning strategy.

    Attributes:
        temperature (float): The temperature parameter for the language model.
        max_tokens (int): The maximum number of tokens to generate.
        llm_class (Type[BaseLLM]): The language model class to use for reasoning.
        usage (str): String describing when it is appropriate to use this reasoning strategy. 
    """
    temperature: float = 0.7
    max_tokens: int = 1500
    llm_class: Type[BaseLLM] = OpenAI
    usage: str

class ReasoningStrategy:
    """Base class for Reasoning Strategies"""
    def __init__(self, config: ReasoningConfig, display: Callable):
        self.llm = config.llm_class(temperature=config.temperature, max_tokens=config.max_tokens) # ign
        self.display = display
        self.usage = config.usage
    def run(self, question):
        raise NotImplementedError()
    
def get_reasoning_config(temperature: float = 0.7) -> ReasoningConfig:
    usage = "This is the default reasoning model that should only be used as a last resort"
    return ReasoningConfig(usage=usage)