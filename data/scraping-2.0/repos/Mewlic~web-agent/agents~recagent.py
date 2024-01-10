import time
from typing import Any, Dict
import openai
from pydantic import BaseModel, Field
from yacs.config import CfgNode
from agents.data import Data
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.experimental.generative_agents.memory import GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)


class RecAgent(GenerativeAgent):
    id: int
    data: Data
    config: CfgNode

    def generate(self, prompt_str: str, prompt_dict: dict) -> str:
        prompt = PromptTemplate.from_template(prompt_str)
        print('\n' + prompt_str + '\n')
        kwargs: Dict[str, Any] = prompt_dict
        generate_result = self.chain(prompt=prompt).run(**kwargs).strip().split("\n")[0]
        print('\n' + generate_result + '\n')
        return generate_result
