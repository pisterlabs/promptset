from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType, AgentExecutor, initialize_agent


class BaseFunctionAgent(object):
    llm: ChatOpenAI
    tools: list[BaseTool]
    callbacks: list[BaseCallbackHandler]

    def __init__(
        self,
        llm,
        tools=None,
        callbacks=None
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks

        if not self.tools:
            self.tools = []

        if not self.callbacks:
            self.callbacks = []

    def add_tool(self, tool: BaseTool):
        self.tools.append(tool)

    def add_callback(self, callback: BaseCallbackHandler):
        self.callbacks.append(callback)

    def get_executor(self, debug: bool = False) -> AgentExecutor:
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            callbacks=self.callbacks,
            return_intermediate_steps=False,
            verbose=debug
        )
