from langchain.chat_models.fireworks import ChatFireworks
from langchain.schema import SystemMessage, HumanMessage

chat = ChatFireworks(model="accounts/fireworks/models/mistral-7b")
system_message = SystemMessage(content="You are to chat with the user.")
human_message = HumanMessage(content="Who are you?")

res = chat([system_message, human_message])
print(res)
