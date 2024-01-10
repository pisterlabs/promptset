# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Agent classes """
import re
from abc import ABC, abstractmethod
from typing import Union, Type, Callable, List

from langchain.agents import (Agent, AgentOutputParser,
                              ConversationalAgent)
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish

from common.models.agent import AgentCapability
from common.utils.http_exceptions import InternalServerError
from common.utils.logging_handler import Logger
from config import LANGCHAIN_LLM
from services.agents.agent_prompts import (PREFIX, PLANNING_PREFIX,
                                           PLAN_FORMAT_INSTRUCTIONS)
from services.agents.agent_tools import (gmail_tool, docs_tool,
                                         calendar_tool, search_tool,
                                         query_tool)

Logger = Logger.get_logger(__file__)

class BaseAgent(ABC):
  """
  Base Agent for LLM Service agents.  All agents are based on Langchain
  agents and basically specify the configuration for a particular variant
  of Langchain agent.
  """

  llm_type: str = None
  """ the LLM Service llm type used to power the agent """

  agent: Agent = None
  """ the langchain agent instance """

  agent_class: Type[Agent] = None
  """ the langchain agent class """

  name:str = None
  """ The name of the agent """

  def __init__(self, llm_type: str):
    self.llm_type = llm_type
    self.agent = None

  @property
  def prefix(self) -> str:
    return PREFIX

  @property
  def format_instructions(self) -> str:
    return FORMAT_INSTRUCTIONS

  @property
  def output_parser_class(self) -> Type[AgentOutputParser]:
    raise NotImplementedError(
        "Derived classes should provide output_parser_class")

  @classmethod
  @abstractmethod
  def capabilities(cls) -> List[str]:
    """ return capabilities of this agent class """

  @abstractmethod
  def get_tools(self) -> List[Callable]:
    """ return tools used by this agent """

  def load_agent(self) -> Agent:
    """ load this agent and return an instance of langchain Agent"""
    tools = self.get_tools()

    llm = LANGCHAIN_LLM.get(self.llm_type)
    if llm is None:
      raise InternalServerError(
          f"Agent: cannot find LLM type {self.llm_type}")

    output_parser = self.output_parser_class()

    self.agent = self.agent_class.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prefix=self.prefix,
        format_instructions=self.format_instructions,
        output_parser=output_parser
    )
    Logger.info(f"Successfully loaded {self.name} agent.")
    Logger.debug(f"prefix=[{self.prefix}], "
                 f"format_instructions=[{self.format_instructions}]")
    return self.agent


class ChatAgent(BaseAgent):
  """
  Chat Agent.  This is an agent configured for basic informational chat with a
  human.  It includes search and query tools.
  """
  def __init__(self, llm_type: str):
    super().__init__(llm_type)
    self.name = "ChatAgent"
    self.agent_class = ConversationalAgent

  @property
  def output_parser_class(self) -> Type[AgentOutputParser]:
    return ToolAgentOutputParser

  @classmethod
  def capabilities(cls) -> List[str]:
    """ return capabilities of this agent class """
    capabilities = [AgentCapability.AGENT_CHAT_CAPABILITY,
                    AgentCapability.AGENT_QUERY_CAPABILITY]
    return capabilities

  def get_tools(self) -> List[Callable]:
    """ return tools used by this agent """
    return [search_tool, query_tool]


class TaskAgent(BaseAgent):
  """
  Task Agent.  This is an agent configured to execute tasks on behalf of a
  human.  Every task has a plan, consisting of plan steps.
  Creation of the plan is done by a planning agent.
  """

  def __init__(self, llm_type: str):
    super().__init__(llm_type)
    self.name = "TaskAgent"
    self.agent_class = ConversationalAgent

  @property
  def output_parser_class(self) -> Type[AgentOutputParser]:
    return ToolAgentOutputParser

  @classmethod
  def capabilities(cls) -> List[str]:
    """ return capabilities of this agent class """
    capabilities = [AgentCapability.AGENT_CHAT_CAPABILITY,
                    AgentCapability.AGENT_QUERY_CAPABILITY,
                    AgentCapability.AGENT_TASK_CAPABILITY]
    return capabilities

  def get_tools(self):
    tools = [gmail_tool, docs_tool, calendar_tool, search_tool, query_tool]
    return tools

  def get_planning_agent(self) -> str:
    """
    This is the agent used by this agent to create plans for tasks.
    """
    return "PlanAgent"


class PlanAgent(BaseAgent):
  """
  Plan Agent.  This is an agent configured to make plans.
  Plans will be executed using a different agent.
  """

  def __init__(self, llm_type: str):
    super().__init__(llm_type)
    self.name = "PlanAgent"
    self.agent_class = ConversationalAgent

  @property
  def prefix(self) -> str:
    return PLANNING_PREFIX

  @property
  def format_instructions(self) -> str:
    return PLAN_FORMAT_INSTRUCTIONS

  @property
  def output_parser_class(self) -> Type[AgentOutputParser]:
    return PlanAgentOutputParser

  @classmethod
  def capabilities(cls) -> List[str]:
    """ return capabilities of this agent class """
    capabilities = [AgentCapability.AGENT_PLAN_CAPABILITY]
    return capabilities

  def get_tools(self):
    tools = [gmail_tool, docs_tool, calendar_tool, search_tool, query_tool]
    return tools


class PlanAgentOutputParser(AgentOutputParser):
  """Output parser for a agent that makes plans."""

  ai_prefix: str = "AI"
  """Prefix to use before AI output."""

  def get_format_instructions(self) -> str:
    return PLAN_FORMAT_INSTRUCTIONS

  def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
    if f"{self.ai_prefix}:" in text:
      return AgentFinish(
          {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
      )
    regex = r"Action: (.*?)[\n]*Action Input: (.*)"
    match = re.search(regex, text)
    if not match:
      # TODO: undo this temporary fix to make the v1 agent terminate
      #raise OutputParserException(
      #    f"MIRA: Could not parse LLM output: `{text}`")
      return AgentFinish(
          {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
      )
    action = match.group(1)
    action_input = match.group(2)
    return AgentAction(action.strip(),
                       action_input.strip(" ").strip('"'), text)

  @property
  def _type(self) -> str:
    return "zero_shot"


class ToolAgentOutputParser(AgentOutputParser):
  """Output parser for a conversational agent that uses tools."""

  ai_prefix: str = "AI"
  """Prefix to use before AI output."""

  def get_format_instructions(self) -> str:
    return FORMAT_INSTRUCTIONS

  def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
    if f"{self.ai_prefix}:" in text:
      return AgentFinish(
          {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
      )
    regex = r"Action: (.*?)[\n]*Action Input: (.*)"
    match = re.search(regex, text)
    if not match:
      # TODO: undo this temporary fix to make the v1 agent terminate
      #raise OutputParserException(
      #    f"MIRA: Could not parse LLM output: `{text}`")
      return AgentFinish(
          {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
      )
    action = match.group(1)
    action_input = match.group(2)
    return AgentAction(action.strip(),
                       action_input.strip(" ").strip('"'), text)

  @property
  def _type(self) -> str:
    return "conversational"

