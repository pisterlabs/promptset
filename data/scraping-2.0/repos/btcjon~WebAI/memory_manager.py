from langchain.memory import ConversationTokenBufferMemory

class ShortTermMemory:
    def __init__(self, size):
        self.size = size
        self.memory = TokenBuffer(max_tokens=size)

    def add_message(self, message):
        self.memory.add_string(message)

    def get_messages(self):
        return self.memory.get_string()
