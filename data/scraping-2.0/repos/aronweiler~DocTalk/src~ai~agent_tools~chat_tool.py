import time
import os
import shared.constants as constants
from shared.selector import get_chat_model, get_llm
from ai.configurations.llm_chain_configuration import LLMChainConfiguration
from langchain.chains import LLMChain as llm_chain
from langchain import PromptTemplate
from ai.agent_tools.utilities.abstract_tool import AbstractTool


class LLMChainTool(AbstractTool):
    def __init__(self, json_args):
        self.configuration = LLMChainConfiguration(json_args)

        if self.configuration.chat_model:
            llm = get_chat_model(
                self.configuration.run_locally, float(self.configuration.ai_temp)
            )
        else:
            llm = get_llm(
                self.configuration.run_locally, float(self.configuration.ai_temp), -1
            )

        self.chain = llm_chain(
            llm=llm,
            verbose=self.configuration.verbose,
            prompt=PromptTemplate.from_template("{inputs}"),
        )

    def run(self, query: str) -> str:
        result = self.chain(inputs=input)

        return result["text"]
