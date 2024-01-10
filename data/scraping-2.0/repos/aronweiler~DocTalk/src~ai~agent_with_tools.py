from shared.selector import get_llm, get_chat_model
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

import re
import json

from ai.abstract_ai import AbstractAI
from ai.ai_result import AIResult

from ai.configurations.react_agent_configuration import AgentWithToolsConfiguration

from ai.agent_tools.utilities.tool_loader import load_tools


class AgentWithTools(AbstractAI):
    def configure(self, json_args) -> None:
        self.configuration = AgentWithToolsConfiguration(json_args)

        if self.configuration.chat_model:
            if self.configuration.run_locally:
                raise Exception(
                    "The chat model can only be used with remote APIs right now."
                )
            router_llm = get_chat_model(
                self.configuration.run_locally,
                ai_temp=float(self.configuration.ai_temp),
            )
            agent_type = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        else:
            router_llm = get_llm(
                self.configuration.run_locally,
                ai_temp=float(self.configuration.ai_temp),
                local_model_path=self.configuration.model,
            )
            agent_type = AgentType.CONVERSATIONAL_REACT_DESCRIPTION

        memory = self._get_memory(router_llm) if self.configuration.use_memory else None

        tools = load_tools(
            config=json_args,
            memory=memory,
            override_llm=None,
        )

        self.agent_chain = initialize_agent(
            tools,
            router_llm,
            agent=agent_type,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": self.configuration.system_message},
        )

    def _get_memory(self, llm):
        memory = ConversationTokenBufferMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )

        return memory

    def query(self, input):
        result = self.agent_chain.run(input=input)

        ai_results = AIResult(result, result)

        return ai_results
