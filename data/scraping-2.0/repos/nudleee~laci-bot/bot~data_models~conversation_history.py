from langchain.memory.chat_memory import BaseChatMemory

class ConversationHistory:
    def __init__(self, memory:  BaseChatMemory = None):
        self._memory = memory