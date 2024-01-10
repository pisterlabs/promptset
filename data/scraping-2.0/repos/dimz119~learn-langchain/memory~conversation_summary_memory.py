from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context({"input": "hi"}, {"output": "whats up"})

print(memory.load_memory_variables({}))
"""
{'history': '\nThe human greets the AI, to which the AI responds.'}
"""

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0), return_messages=True)
memory.save_context({"input": "hi"}, {"output": "whats up"})

# print(memory.load_memory_variables({}))
"""
{'history': [SystemMessage(content='\nThe human greets the AI, to which the AI responds.')]}
"""

# Directly use the memory
messages = memory.chat_memory.messages
"""
[HumanMessage(content='hi'), AIMessage(content='whats up')]
"""
previous_summary = ""
print(memory.predict_new_summary(messages, previous_summary))
"""
The human greets the AI, to which the AI responds.
"""

# Initializing with messages/existing summary
history = ChatMessageHistory()
history.add_user_message("hi")
history.add_ai_message("hi there!")

memory = ConversationSummaryMemory.from_messages(
    llm=OpenAI(temperature=0),
    chat_memory=history,
    return_messages=True
)
print(memory.buffer)
"""
The human greets the AI, to which the AI responds with a friendly greeting.
"""

# summary initialization
memory = ConversationSummaryMemory(
    llm=OpenAI(temperature=0),
    buffer="The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.",
    chat_memory=history,
    return_messages=True
)
print(memory.buffer)
"""
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
"""