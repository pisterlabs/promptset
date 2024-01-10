from typing import Dict, List

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.agents.tools import BaseTool

from .builder import AgentBuilder

class AgentManager:
    def __init__(
        self,
        toolsets: list[BaseTool] = [],
    ):
        self.toolsets: list[BaseTool] = toolsets
        self.memories: Dict[str, BaseChatMemory] = {}
        self.executors: Dict[str, AgentExecutor] = {}

    def create_memory(self) -> BaseChatMemory:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_or_create_memory(self, key: str) -> BaseChatMemory:
        if not (key in self.memories):
            self.memories[key] = self.create_memory()
        return self.memories[key]
    
    def create_executor(self, key: str, callbacks: List[BaseCallbackHandler] = []) -> AgentExecutor:
        builder = AgentBuilder(self.toolsets)
        builder.build_parser()

        callback_manager = CallbackManager(callbacks)

        builder.build_llm(callback_manager)
        builder.build_global_tools()
        builder.build_parser()

        memory: BaseChatMemory = self.get_or_create_memory(key)

        tools = [
            *builder.get_global_tools(),
            *self.toolsets,
        ]
        
        for tool in tools:
            tool.callback_manager = callback_manager

        executor = AgentExecutor.from_agent_and_tools(
            agent=builder.get_agent(),
            tools=tools,
            memory=memory,
            callback_manager=callback_manager,
            verbose=True,
        )

        self.executors[key] = executor
        return executor

    def get_or_create_executor(self, key: str) -> AgentExecutor:
        if not (key in self.executors):
            self.executors[key] = self.create_executor(key=key)
        return self.executors[key]

    def remove_executor(self, key: str) -> None:
        if key in self.executors:
            del self.executors[key]

    @staticmethod
    def create(toolsets: List[BaseTool]) -> "AgentManager":
        return AgentManager(
            toolsets=toolsets,
        )
