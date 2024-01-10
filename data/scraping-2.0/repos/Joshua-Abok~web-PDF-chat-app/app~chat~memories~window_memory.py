from langchain.memory import ConversationBufferMemory
from app.chat.memories.histories.sql_history import SqlMessageHistory


# make builder function -> gonna be assigned in the memory_map
def window_buffer_memory_builder(chat_args): 
    return ConversationBufferMemory(
        memory_key="chat_history", 
        output_key="answer", 
        return_messages=True, 
        chat_memory=SqlMessageHistory(
            conversation_id=chat_args.conversation_id
        ),
        k=2  # when fetching the memory, how many xchanges to include
    )