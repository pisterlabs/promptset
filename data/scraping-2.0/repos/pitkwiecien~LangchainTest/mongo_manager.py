import os

from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory


class MongoManager:
    __instance = None

    def __init__(self, session_id):
        self.session_id = session_id

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(MongoManager, cls).__new__(cls)
        return cls.__instance

    def get_memory(self):
        api_key = os.environ["MONGODB_URI"]
        print(api_key)
        message_history = MongoDBChatMessageHistory(
            connection_string=api_key,
            session_id=self.session_id
        )

        return message_history
