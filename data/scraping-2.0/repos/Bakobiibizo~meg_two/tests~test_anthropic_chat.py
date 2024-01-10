import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_services.anthropic_chat import AnthropicChatBot
from langchain.schema import HumanMessage, SystemMessage

def test_anthropic_chat():
    chat_bot = AnthropicChatBot()
    messages=[
    SystemMessage(content="You a friendly and helpful chatbot. You provide verbose and detailed answers to user queries. If you do not know the answer to something you simpily say that you do not know."),
    HumanMessage(content="Hello, how are you?")
    ]
    print(messages)
    response = chat_bot.get_chat_from_message(message=messages)
    print(response)

if __name__ == "__main__":
    test_anthropic_chat()