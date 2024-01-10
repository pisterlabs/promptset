from langchain.memory import ConversationBufferMemory

def setup_memory(memory_key="chat_history"):
    return ConversationBufferMemory(memory_key=memory_key)

