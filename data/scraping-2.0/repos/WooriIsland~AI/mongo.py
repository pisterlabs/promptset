from pymongo import MongoClient

mongodb_URI = "mongodb://localhost:27017/"
client = MongoClient(mongodb_URI)
print(client.list_database_names())

###

from langchain.memory import MongoDBChatMessageHistory

message_history = MongoDBChatMessageHistory(
    connection_string=mongodb_URI, session_id="test-session"
)

message_history.add_user_message("hi!")
message_history.add_ai_message("whats up?")

print(message_history.messages)