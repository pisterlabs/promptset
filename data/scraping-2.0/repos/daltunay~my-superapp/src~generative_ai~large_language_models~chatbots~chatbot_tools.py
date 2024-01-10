import typing as t
from functools import cached_property

from langchain.agents import (AgentExecutor, AgentType, initialize_agent,
                              load_tools)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import BaseTool

from src.generative_ai.large_language_models.chatbots import Chatbot, ModelArgs


class ChatbotTools(Chatbot):
    available_tools = ["google-search", "arxiv", "wikipedia", "stackexchange", "human"]

    def __init__(
        self,
        tool_names: t.List[str] | None = None,
        **model_kwargs: t.Unpack[ModelArgs],
    ) -> None:
        super().__init__(**model_kwargs)
        self.tool_names = tool_names or []
        self.memory.input_key = "input"

    # @property
    # def callbacks(self) -> t.List[BaseCallbackHandler]:
    #     return [super().callbacks[1]]

    @cached_property
    def tools(self) -> t.List[BaseTool]:
        return load_tools(tool_names=self.tool_names)

    @staticmethod
    def update_agent_prompt_template(
        agent: AgentExecutor,
        text: str,
        input_variable: str | None = None,
    ):
        template = agent.agent.llm_chain.prompt.template
        newline_index = agent.agent.llm_chain.prompt.template.find("\n\n")
        agent.agent.llm_chain.prompt.template = text + template[newline_index:]
        if input_variable:
            agent.agent.llm_chain.prompt.input_variables.append(input_variable)
        return agent

    @cached_property
    def chain(self) -> AgentExecutor:
        agent = initialize_agent(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "input_variables": [
                    "input",
                    "chat_history",
                    "agent_scratchpad",
                    "language",
                ]
            },
            tools=self.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
        agent = self.update_agent_prompt_template(
            agent=agent,
            text="Assistant is a large language model, speaking in {language}.",
            input_variable="language",
        )
        return agent

    def ask(
        self,
        query: str,
        language: str | None = None,
    ) -> str:
        return self.chain.run(
            input=query,
            language=language or "the input language",
            callbacks=self.callbacks,
        )
