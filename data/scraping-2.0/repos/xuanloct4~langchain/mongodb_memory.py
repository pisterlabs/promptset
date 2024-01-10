import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

# Provide the connection string to connect to the MongoDB database
# connection_string = "mongodb://root:example@mongo:27017"
connection_string = "mongodb://root:example@localhost:27017"
from langchain.memory import MongoDBChatMessageHistory

message_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id="test-session"
    )

# message_history.add_user_message("hi!")
# message_history.add_ai_message("whats up?")
print(message_history.messages)