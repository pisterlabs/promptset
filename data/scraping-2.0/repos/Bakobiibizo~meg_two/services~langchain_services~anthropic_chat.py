import os
from langchain.chat_models import ChatAnthropic
from services.context_window import ContextWindow
from dotenv import load_dotenv

load_dotenv()

class AnthropicChatBot():
    def __init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.chat = ChatAnthropic(anthropic_api_key=self.anthropic_api_key,model="claude-v1-100k", max_tokens_to_sample=40000)
        self.context = ContextWindow()

    def get_chat_from_messages(self, messages):
        response = self.chat.generate(messages=messages)
        print(response)

    def get_chat_from_message(self, message):
        response = self.chat(messages=message)
        print(response)