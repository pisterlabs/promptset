from abc import ABC, abstractmethod
from enum import Enum
import json

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import Runnable
from .multi_agent_prompts import base_prompt
from .agent_input import Agent_Input

from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelInput,
)

class agent_type(ABC):
    UNDEFINED = "UNDEFINED"
    CONTROLLER = "controller"

class base_agent:
    def __init__(self, type : agent_type, llm : BaseLanguageModel) -> None:
        self.llm = llm
        self.prompt = base_prompt
        self.type = type

    @abstractmethod
    def run(self, input: Agent_Input):
        pass

class defined_agent(base_agent):
    def __init__(self, llm, type: agent_type, prompt: str = None) -> None:
        super().__init__(llm, type)
        self.prompt = prompt

class controller_agent(defined_agent):
    def __init__(self, llm):
        super().__init__(llm, agent_type.CONTROLLER)
