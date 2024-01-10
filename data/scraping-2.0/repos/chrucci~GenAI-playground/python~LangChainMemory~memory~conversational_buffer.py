from langchain.memory import ConversationBufferMemory


def load_mem(input, output):
    memory = ConversationBufferMemory()
    memory.save_context(input, output)
    return memory.load_memory_variables({})

def load_mem_return(input, output):
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context(input, output)
    return memory.load_memory_variables({})