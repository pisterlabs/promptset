import time
from typing import Callable, Optional

from langchain.agents import AgentExecutor

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool

from src.core.assistant.consts import MAX_HISTORY_TIME

from langchain.memory import ConversationBufferWindowMemory


def identity_function(x):
    return x


class ChatBotTemplate:
    def __init__(
        self,
        main_llm: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
        format_function: Callable = identity_function,
        tool_format_function: Callable = identity_function,
        output_parser: Callable = identity_function,
    ):
        self.tools = tools
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a home assistant, your goal is to listen to user orders. "
                        "Be creative when performing tasks and use your own knowledge."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{human_input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if tools:
            main_llm = self.bind_tools(
                main_llm, tools=tools, format_function=tool_format_function
            )

        self.agent = (
            {
                "human_input": lambda x: x["human_input"],
                "agent_scratchpad": lambda x: format_function(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | main_llm
            | output_parser()
        )
        self.last_history_timestamp: Optional[float] = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=10, return_messages=True
        )

    def chat(self, human_input: str):
        agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, memory=self.memory, verbose=True
        )
        history_timestamp = time.time()
        if (
            self.last_history_timestamp
            and history_timestamp - self.last_history_timestamp > MAX_HISTORY_TIME
        ):
            self.clear_chat_history()
        self.last_history_timestamp = history_timestamp

        output = agent_executor.invoke({"human_input": human_input})

        return output["output"]

    def clear_chat_history(self):
        self.memory.clear()

    def bind_tools(
        self,
        llm: BaseChatModel,
        tools: Optional[list[BaseTool]],
        format_function: Callable,
    ) -> Runnable:
        return llm.bind(functions=[format_function(tool) for tool in tools])
