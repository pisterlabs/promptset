from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

from oslui.agent.base import BaseAgent


class TranslateAgent(BaseAgent):
    """
        TranslateAgent: translate natural language command to shell command
    """
    chain: LLMChain = None

    def __init__(self, llm: BaseLLM, prompt: PromptTemplate = None, memory: PromptTemplate = None):
        super().__init__(llm, prompt, memory)
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, input_msg: str) -> str:
        return self.chain.run(input_msg)
