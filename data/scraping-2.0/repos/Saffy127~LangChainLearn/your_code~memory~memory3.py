import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, message_to_dict

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

dicts = messages_to_dict(histroy.messages)

