from pydantic import BaseModel
from langchain.llms.base import BaseLLM
from langchain.llms import OpenAI
from typing import Type

class LLMChainConfig(BaseModel):
    """
    A configuration class for the chain strategy.

    Attributes:
        temperature (float): The temperature parameter for the language model.
        max_tokens (int): The maximum number of tokens to generate.
        llm_class (Type[BaseLLM]): The language model class to use for reasoning.
        usage (str): String describing when it is appropriate to use this chain strategy. 
    """
    temperature: float = 0.7
    max_tokens: int = 1500
    llm_class: Type[BaseLLM] = OpenAI # Overrideable default
    usage: str