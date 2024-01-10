from typing import List

# Import necessary modules and classes
from langchain.agents import initialize_agent, load_tools, AgentType, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Import models and settings from Django app
from apps.llm.conversation_messages import ConversationMessageRepository
from apps.llm.models import SenderMessage, ConversationMessage
from django_llm import settings


# Define a class for creating agents
class AgentFactory:
    def __init__(self):
        # Initialize the chat message repository
        self.chat_message_repository = ConversationMessageRepository()

    async def create_agent(
            self,
            tool_names: List[str],
            chat_id: str = None,
            streaming=False,
            callback_handlers: List[BaseCallbackHandler] = None,
    ) -> AgentExecutor:
        # Initialize a ChatOpenAI instance with required parameters
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            callbacks=callback_handlers,
        )

        # Load tools using the tool names and the ChatOpenAI instance
        tools = load_tools(tool_names, llm=llm)

        # Load agent memory based on the chat_id (if provided)
        memory = await self._load_agent_memory(chat_id)

        # Initialize an agent with the specified parameters
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

    async def _load_agent_memory(
            self,
            chat_id: str = None,
    ) -> ConversationBufferMemory:
        # If no chat_id is provided, create an empty ConversationBufferMemory
        if not chat_id:
            return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create a ConversationBufferMemory with the provided chat_id
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Retrieve chat messages associated with the provided chat_id
        chat_messages: List[ConversationMessage] = await self.chat_message_repository.get_chat_messages(chat_id)

        # Populate memory with user and AI messages from the chat
        for message in chat_messages:
            if message.sender == SenderMessage.USER.value:
                memory.chat_memory.add_user_message(message.content)
            elif message.sender == SenderMessage.AI.value:
                memory.chat_memory.add_ai_message(message.content)

        return memory
