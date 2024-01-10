from chatbot_settings import ChatBotSettings
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

chatbotSettings = ChatBotSettings()


messages = [
            SystemMessage(content="You are scientific Chatbot inspired by how Feynman, Karl Popper, and Carl Sagan think and talk."),
]

while True:
    inquiry = input("Feyopperagan Chatbot- Let's Chat: ")
    if inquiry.lower() == 'exit':
       print("Goodbye!")
       break

    messages.append(HumanMessage(content=inquiry))
    chatOpenAI = ChatOpenAI()
    response= chatOpenAI(messages)
    print(response.content)