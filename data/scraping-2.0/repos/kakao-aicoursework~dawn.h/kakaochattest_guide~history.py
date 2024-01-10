import os
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

CUR_DIR = os.path.dirname(os.path.abspath('./kakaochattest_guide'))
HISTORY_DIR = os.path.join(CUR_DIR, "chat_histories")


class History:
    def __init__(self, conversation_id: str = 'fa1010'):

        self.history = self.load_conversation_history(conversation_id)

    def log_user_message(self, user_message: str):
        self.history.add_user_message(user_message)

    def log_bot_message(self, bot_message: str):
        self.history.add_ai_message(bot_message)

    def get_chat_history(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            chat_memory=self.history,
        )

        return memory.buffer

    @staticmethod
    def load_conversation_history(conversation_id: str):
        file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
        return FileChatMessageHistory(file_path)
