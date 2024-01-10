from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict
from langchain.schema import messages_from_dict

history = ChatMessageHistory()

history.add_user_message("AIとは何？")
history.add_ai_message("AIとは、人工知能のことです。")

message_list = history.messages
message_dict = messages_to_dict(message_list)
print(message_list)
print(message_dict)

message_list_from_dict = messages_from_dict(message_dict)
print(message_list_from_dict)
