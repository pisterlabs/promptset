"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import re
from typing import List

from kink import di
from langchain.agents import AgentType, initialize_agent
from langchain_core.pydantic_v1 import BaseModel, Field

from .. import MedPrompter


# from langchain_core.prompts import MessagesPlaceholder
# from langchain.memory.buffer import ConversationBufferMemory
class BaseMedpromptAgent:

    class AgentInput(BaseModel):
        """Chat history with the bot."""
        current_patient_context: str
        input: str

    def __init__(
        self,
        llm = None,
        input_type: BaseModel = None,
        template_path=None,
        prefix=None,
        suffix=None,
        tools: List = [],
    ):
        _name = self.get_name()
        self.llm = llm
        if llm is None:
                self.llm = di["main_llm"]
        if prefix is None:
            prefix = _name + "_prefix_v1.jinja"
        if suffix is None:
            suffix = _name + "_suffix_v1.jinja"
        self.med_prompter = MedPrompter()
        self.prefix = prefix
        # If prefix is a jinja template, generate the prompt
        if "jinja" in prefix:
            self.med_prompter.set_template(template_path=template_path, template_name=prefix)
            self.prefix = self.med_prompter.generate_prompt()
        self.suffix = suffix
        if "jinja" in suffix:
            self.med_prompter.set_template(template_path=template_path, template_name=suffix)
            self.suffix = self.med_prompter.generate_prompt()
        self.tools = tools
        # current_patient_context = MessagesPlaceholder(variable_name="current_patient_context")
        # memory = ConversationBufferMemory(memory_key="current_patient_context", return_messages=True)
        self.agent_kwargs = {
            "prefix": self.prefix,
            "suffix": self.suffix,
            # "memory_prompts": [current_patient_context],
            "input_variables": ["input", "agent_scratchpad", "current_patient_context"]
        }
        if input_type is None:
            self.input_type = self.AgentInput
        else:
            self.input_type = input_type


    def get_agent(self):
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            stop=["\nObservation:"],
            max_iterations=len(self.tools) + 3,
            handle_parsing_errors=True,
            agent_kwargs=self.agent_kwargs,
            verbose=True).with_types(input_type=self.input_type)

    def get_name(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()