from string import Template
from typing import List, Any

from pydantic import BaseModel

from morpher.llm import OpenAIWrapper
from morpher.prompts import CALCULATOR_TEMPLATE, DEFAULT_TEMPLATE
from morpher.tools import Tool


class MiscTools(BaseModel):
    tool_info: List[Tool] = []

    def __init__(self, **data: Any):
        super().__init__(**data)

        self.tool_info: List[Tool] = [
            Tool(
                name="meaning_of_life",
                description="Useful to get information on an unknown topic by searching it online. Input must be a string.",
                func=self.meaning_of_life
            ),
            Tool(
                name="calculator",
                description="Useful to perform basic arithmetic operations. Input must be a string.",
                func=self.calculator
            ),
        ]

    @staticmethod
    def meaning_of_life():
        """
        Mock tool to return a static response.

        :return: Static string.
        """
        return "42 is the answer to life, the universe, and everything."

    @staticmethod
    def calculator(input: str):
        """
        Calculates simple arithmetic operations using an LLM.

        :param input: Query to calculate.
        :return: Answer of the query as a string.
        """
        prompt_template = Template(CALCULATOR_TEMPLATE)
        prompt = prompt_template.substitute(task=input)
        result = OpenAIWrapper.generate(prompt)
        return result

    # @staticmethod
    # def default_tool(input: str):
    #     """
    #     Default tool to improve fault tolerance.
    #
    #     :param input: Query as a string.
    #     :return: Answer to the query.
    #     """
    #     prompt_template = Template(DEFAULT_TEMPLATE)
    #     prompt = prompt_template.substitute(task=input, memory=get_memory())
    #     result = OpenAIWrapper.generate(prompt)
    #     return result
