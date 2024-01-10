# https://www.youtube.com/watch?v=2xxziIWmaSA&t=1625s
import os

from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

history = ChatMessageHistory()

history.add_ai_message("hi!")

history.add_user_message("what is the capital of Norway?")

print(history.messages)

ai_response = chat(history.messages)
print(ai_response)

history.add_ai_message(ai_response.content)
print(history.messages)
