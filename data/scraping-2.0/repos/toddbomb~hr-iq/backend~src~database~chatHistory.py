from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history",  return_messages=True)