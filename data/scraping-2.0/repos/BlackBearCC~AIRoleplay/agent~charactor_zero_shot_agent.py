# charactor_zero_shot_agent.py
from dataclasses import Field
from typing import Sequence, Optional, List, Union, Any, Tuple
from langchain.agents import ZeroShotAgent, AgentOutputParser, BaseSingleActionAgent
from langchain.agents.xml.prompt import agent_instructions
from langchain.callbacks.base import Callbacks
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate

from agent.CustomOutputParser import CustomOutputParser
from agent.charactor_agent_prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX, TSET

class CharactorZeroShotAgent(BaseSingleActionAgent):
    """Agent that uses XML tags.

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:

        .. code-block:: python

            from langchain.agents import XMLAgent
            from langchain

            tools = ...
            model =


    """

    tools: List[BaseTool]
    """List of tools this agent has access to."""
    llm_chain: LLMChain
    """Chain to use to predict action."""

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            TSET
        ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")

    @staticmethod
    def get_default_output_parser() -> CustomOutputParser:
        return CustomOutputParser()

    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = self.llm_chain(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]

    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = await self.llm_chain.acall(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]


    # @property
    # def llm_prefix(self) -> str:
    #     return "思考:"
    # @property
    # def observation_prefix(self) -> str:
    #     return "观察:"
    #
    # @classmethod
    # def create_prompt(
    #     cls,
    #     tools: Sequence[BaseTool],
    #     prefix: str = "提示前缀",
    #     suffix: str = "提示后缀",
    #     format_instructions: str = "你的自定义格式指令",
    #     input_variables: Optional[List[str]] = None,
    # ) -> PromptTemplate:
    #     """
    #     创建一个自定义风格的提示模板。
    #
    #     Args:
    #         tools: 代理将访问的工具列表，用于格式化提示。
    #         prefix: 放在工具列表前的字符串。
    #         suffix: 放在工具列表后的字符串。
    #         format_instructions: 自定义格式指令。
    #         input_variables: 最终提示将期望的输入变量列表。
    #
    #     Returns:
    #         组装好的提示模板。
    #     """
    #     # 生成工具描述字符串
    #     tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    #     # 生成工具名称列表
    #     tool_names = ", ".join([tool.name for tool in tools])
    #     # 格式化指令
    #     format_instructions = format_instructions.format(tool_names=tool_names)
    #     # 组装模板
    #     template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
    #     # 设置输入变量
    #     if input_variables is None:
    #         input_variables = ["input", "agent_scratchpad"]
    #     return PromptTemplate(template=template, input_variables=input_variables)
