import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_services.openai_chat import OpenAIChatBot
from langchain.schema import HumanMessage

def test_openai_chat():
    chat_bot = OpenAIChatBot(model="ft-IIMcvAS1EgQ47FFgEzCOorA4")
    messages = [
        HumanMessage(
        content="write a poem in the style of charles bukowski about a duck finding a fedora"
        )
    ]
    print(messages)
    response = chat_bot.get_chat_response(messages=messages)
    print(response)

if __name__ == "__main__":
    test_openai_chat()