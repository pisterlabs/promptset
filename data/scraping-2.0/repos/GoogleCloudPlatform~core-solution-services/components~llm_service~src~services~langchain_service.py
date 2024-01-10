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

""" Langchain service """

import inspect
from typing import Optional, Any, List
from common.models import UserChat
from common.models.agent import AgentType
from common.utils.errors import ResourceNotFoundException
from common.utils.http_exceptions import InternalServerError
from common.utils.logging_handler import Logger
import langchain.agents as langchain_agents
from langchain.schema import HumanMessage, AIMessage
from config import LANGCHAIN_LLM, CHAT_LLM_TYPES

Logger = Logger.get_logger(__file__)

async def langchain_llm_generate(prompt: str, llm_type: str,
                                 user_chat: Optional[UserChat] = None):
  """
  Use langchain to generate text with an LLM given a prompt.  This is
    always done asynchronously, and so must be used in a route defined with
    async def.

  Args:
    prompt: the text prompt to pass to the LLM

    llm_type: the type of LLM to use (default to openai)

    user_chat (optional): a user chat to use for context

  Returns:
    the text response.
  """
  Logger.info(f"Generating text with langchain llm_type {llm_type}")
  try:
    # get LLM object
    llm = LANGCHAIN_LLM.get(llm_type)
    if llm is None:
      raise ResourceNotFoundException(f"Cannot find llm type '{llm_type}'")

    if llm_type in CHAT_LLM_TYPES:
      # use langchain chat interface for openai

      # create msg history for user chat if it exists
      if user_chat is not None:
        msg = langchain_chat_history(user_chat)
      else:
        msg = []
      msg.append(HumanMessage(content=prompt))

      Logger.info(f"generating text for [{prompt}]")
      response = await llm.agenerate([msg])
      response_text = response.generations[0][0].message.content
      Logger.info(f"response {response_text}")
    else:
      msg = []
      if user_chat is not None:
        msg = user_chat.history
      msg.append(prompt)
      response = await llm.agenerate(msg)
      response_text = response.generations[0][0].text

    return response_text
  except Exception as e:
    raise InternalServerError(str(e)) from e


def get_model(llm_type: str) -> Any:
  """ return a langchain model given type """
  llm = LANGCHAIN_LLM.get(llm_type)
  return llm


def langchain_class_from_agent_type(agent_type: AgentType):
  """ get langchain agent class object from agent type """
  agent_class_name = agent_type.value.split("langchain_")[1] + "Agent"
  agent_classes = inspect.getmembers(langchain_agents, inspect.isclass)
  agent_class = [cpair[1] for cpair in agent_classes
                    if cpair[0] == agent_class_name][0]
  return agent_class


def langchain_chat_history(user_chat: UserChat) -> List:
  """ get langchain message history from UserChat """
  langchain_history = []
  for entry in user_chat.history:
    content = UserChat.entry_content(entry)
    if user_chat.is_human(entry):
      langchain_history.append(HumanMessage(content=content))
    elif user_chat.is_ai(entry):
      langchain_history.append(AIMessage(content=content))
  return langchain_history
