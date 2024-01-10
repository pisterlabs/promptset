from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: BaseChatModel,
        prefix: str | None = None,
    ) -> None:
        """
        Create dialogue agent.

        Args:
            name (str): Agent name.
            system_message (SystemMessage): System message to define agent roles.
            model (BaseChatModel): LLM.
            prefix (str | None, optional):
                Prefix used in chat. Defaults to None.
                If `prefix` is None, set "Agent name: " as prefix.
        """
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = prefix if prefix is not None else f"{self.name}: "
        self.reset()

    def reset(self) -> None:
        """
        Reset message history.
        """
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chat model to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: BaseChatModel,
        tools: List[BaseTool],
        prefix: str | None = None,
    ) -> None:
        """
        Create dialogue agent using tools.

        Args:
            name (str): Agent name.
            system_message (SystemMessage): System message to define agent roles.
            model (BaseChatModel): LLM.
            tools (List[BaseTool]): Tools.
            prefix (str | None, optional):
                Prefix used in chat. Defaults to None.
                If `prefix` is None, set "Agent name: " as prefix.
        """
        super().__init__(name, system_message, model, prefix=prefix)
        self.tools = tools

    def send(self) -> str:
        """
        Applies the chat model to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content
