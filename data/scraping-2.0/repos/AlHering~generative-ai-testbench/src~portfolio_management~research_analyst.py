# -*- coding: utf-8 -*-
"""
****************************************************
*     generative_ai_testbench:portfolio_management                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import List, Any
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain


class ResearchAnalyst(BaseMultiActionAgent):
    """
    Class, representing a Research Analyst Agent.

    Responsiblities: Research analysts conduct in-depth research on financial markets, sectors, and individual securities. 
    They analyze economic trends, financial statements, and company performance to identify investment opportunities and provide recommendations to 
    the portfolio manager.

    Modelled responsiblities:
    - 
    """

    def __init__(self, general_llm: Any, math_llm: Any = None) -> None:
        """
        Initiation method.
        :param general_llm: General LLM.
        :param math_llm: Optional Math LLM.
        """
        self.general_llm = general_llm
        self.math_llm = general_llm if math_llm is None else math_llm
        self.tools = self._initiate_tools()

    """
    Tools
    """

    def _initiate_tools(self) -> List[Tool]:
        """
        Internal method for initiating tools.
        """
        return [
            self._initiate_general_llm_tool,
            self._initiate_math_llm_tool
        ]

    def _initiate_general_llm_tool(self) -> Tool:
        """
        Internal method for initiating general LLM tool.
        """
        promt_template = PromptTemplate(
            input_variables=["input"],
            template="{input}"
        )
        llm_chain = LLMChain(llm=self.general_llm, prompt=promt_template)

        return Tool(
            name="General Language Model",
            func=llm_chain.run,
            description="Use this tool for general purpose question answering and logic."
        )

    def _initiate_math_llm_tool(self) -> Tool:
        """
        Internal method for initiating math LLM tool.
        """
        return Tool(
            name="Calculator",
            func=LLMMathChain(llm=self.self.math_llm).run,
            description="Use this tool for questions which involve math or calculations."
        )
