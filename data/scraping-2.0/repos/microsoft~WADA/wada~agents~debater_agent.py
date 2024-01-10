# Copyright © Microsoft Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Union

from colorama import Fore
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    ConversationalAgent,
    Tool,
)
from langchain.chains import ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

from wada.generators import DebatePromptTemplateGenerator
from wada.typing import ModelType


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={
                    "output": llm_output.split("Final Answer:")[-1].strip()
                },
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action,
                           tool_input=action_input.strip(" ").strip('"'),
                           log=llm_output)


messages = [
    SystemMessagePromptTemplate.from_template(
        DebatePromptTemplateGenerator().get_sys_msg_prompt_template()),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(
        DebatePromptTemplateGenerator().get_hmn_msg_prompt_template()),
]

prompt = ChatPromptTemplate.from_messages(messages)

# tools = load_tools(["arxiv", "ddg-search", "wikipedia"])

tools = [
    Tool(
        name="Search", func=DuckDuckGoSearchRun().run, description=
        "A search engine. Useful for when you need to answer questions about current events. "
        "Input should be a search query"),
    Tool(
        name="Wikipedia", func=WikipediaAPIWrapper().run, description=
        "An online encyclopedia providing information on various topics. Useful for research "
        "and general knowledge. Input should be a search query"),
    Tool(
        name="Arxiv", func=ArxivAPIWrapper().run, description=
        "An online repository of scientific papers. Useful for accessing research papers in "
        "various fields. Input should be a search query")
]


class DebaterAgent:
    r"""A langchain conversational agent that can debate.

    Args:
        tools (list[Tool]): The tools to use for the agent.
            (default: :obj:`List[Tool]` loaded from :obj:`tools`)
        prompt (ChatPromptTemplate): The prompt template to use for the agent.
            (default: :obj:`ChatPromptTemplate` loaded from :obj:`prompt`)
        temperature (float): The temperature to use for the agent.
            (default: :obj:`0.0`)
        model (ModelType): The model type to use for the agent.
            (default: :obj:`ModelType.GPT_4`)
        sys_msg_dict (dict[str, str]): A dictionary of system messages to use
            for the agent. (default: :obj:`{}`)
        verbose (bool): Whether or not to print the agent's output.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        tools: list[Tool] = tools,
        prompt: ChatPromptTemplate = prompt,
        temperature: float = 0.0,
        model: ModelType = ModelType.GPT_4,
        sys_msg_dict: dict[str, str] = {},
        verbose: bool = False,
    ) -> None:
        self.tools = tools
        self.tool_names = [tool.name for tool in self.tools]
        self.prompt = prompt
        self.output_parser = CustomOutputParser()
        self.sys_msg_dict = sys_msg_dict
        self.model = model

        self.chat = AzureChatOpenAI(
            temperature=temperature,
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            deployment_name=self.model.value,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type="azure",
        )

        self.memory = ConversationBufferMemory(return_messages=True,
                                               memory_key="chat_history",
                                               input_key="input", verbose=True)

        self.conversation = ConversationChain(memory=self.memory,
                                              prompt=prompt, llm=self.chat,
                                              verbose=True)

        self.agent = ConversationalAgent(llm_chain=self.conversation,
                                         output_parser=self.output_parser,
                                         stop=["\nObservation:"],
                                         allowed_tools=self.tool_names,
                                         verbose=True)

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=verbose)

    def step(self, input: str) -> str:
        res = self.agent_executor.run(input=input, tool_names=self.tool_names,
                                      **self.sys_msg_dict)
        return res.replace("###", "")

    def reset(self) -> None:
        self.memory.clear()

    def print_memory(self):
        for i in range(len(self.memory.buffer)):
            msg = self.memory.buffer[i]
            if isinstance(msg, HumanMessage):
                print(Fore.YELLOW + "*** Human:\n" + Fore.WHITE + msg.content +
                      "\n")
            elif isinstance(msg, AIMessage):
                print(Fore.YELLOW + "*** AI:\n" + Fore.WHITE + msg.content +
                      "\n")
