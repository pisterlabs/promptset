from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("what's up?")

print(memory.load_memory_variables({}))
"""
{'history': "Human: hi!\nAI: what's up?"}
"""

# memory key change
memory = ConversationBufferMemory(memory_key="chat_history")
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("what's up?")

print(memory.load_memory_variables({}))
"""
{'chat_history': "Human: hi!\nAI: what's up?"}
"""

# return in str type or list type
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("what's up?")

print(memory.load_memory_variables({}))
"""
{'history': [HumanMessage(content='hi!'), AIMessage(content="what's up?")]}
"""