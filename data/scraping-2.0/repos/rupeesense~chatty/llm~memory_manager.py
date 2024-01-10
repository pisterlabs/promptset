from langchain.memory import ConversationBufferWindowMemory


#
# Let LLM manage the memory as well.
# This is the memory manager.
#
class ChatHistory:
    def __init__(self):
        self.user_chat_history = {}
        self.lookback = 10
        pass

    def get_chat_history(self, user_id: str) -> str:
        if user_id in self.user_chat_history:
            return self.user_chat_history[user_id].load_memory_variables({})['history']
        else:
            self.user_chat_history[user_id] = ConversationBufferWindowMemory(k=self.lookback)
            return ""

    def update_chat_history(self, user_id: str, user_message: str, bot_message: str):
        if user_id in self.user_chat_history:
            self.user_chat_history[user_id].save_context({'input': user_message}, {'output': bot_message})
        else:
            self.user_chat_history[user_id] = ConversationBufferWindowMemory(k=self.lookback)
            self.user_chat_history[user_id].save_context({'input': user_message}, {'output': bot_message})
