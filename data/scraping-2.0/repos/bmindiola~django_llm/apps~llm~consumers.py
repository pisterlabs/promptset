import json
import os
import django

# Import necessary modules and classes
from apps.llm.agents.agent_factory import AgentFactory
from apps.llm.agents.callbacks import AsyncStreamingCallbackHandler
from apps.llm.conversation_messages import ConversationMessageRepository

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

# Import the WebSocket consumer class
from channels.generic.websocket import AsyncWebsocketConsumer
from langchain.agents import AgentExecutor

from apps.llm.models import SenderMessage


# Define a WebSocket consumer for chat functionality
class ConversationConsumer(AsyncWebsocketConsumer):
    agent: AgentExecutor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_factory = AgentFactory()
        self.chat_message_repository = ConversationMessageRepository()

    async def connect(self):
        # Get the chat_id from the URL route
        chat_id = self.scope['url_route']['kwargs'].get('chat_id')

        # Create an agent with the specified parameters
        self.agent = await self.agent_factory.create_agent(
            tool_names=["llm-math"],
            chat_id=chat_id,
            streaming=True,
            callback_handlers=[AsyncStreamingCallbackHandler(self)],
        )

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        chat_id = text_data_json['chat_id']

        # Send the user's message to the agent for processing
        response = await self.message_agent(message, chat_id)

        # Send the agent's response back to the WebSocket client
        await self.send(text_data=json.dumps({'message': response, 'type': 'answer'}))

    async def message_agent(self, message: str, chat_id: str):
        # Save the user's message to the repository
        await self.chat_message_repository.save_message(message=message, sender=SenderMessage.USER.value,
                                                        chat_id=chat_id)

        # Use the agent to process the user's message and get a response
        response = await self.agent.arun(message)

        # Save the agent's response to the repository
        await self.chat_message_repository.save_message(message=response, sender=SenderMessage.AI.value,
                                                        chat_id=chat_id)

        return response

    def my_callback(self, message):
        print("Callback received:", message)
