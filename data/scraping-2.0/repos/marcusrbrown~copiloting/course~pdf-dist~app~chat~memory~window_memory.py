from langchain.memory import ConversationBufferWindowMemory

from app.chat.memory.history.sql_history import SqlMessageHistory
from app.chat.models import ChatArgs


def build_window_buffer_memory(chat_args: ChatArgs):
    return ConversationBufferWindowMemory(
        chat_memory=SqlMessageHistory(conversation_id=chat_args.conversation_id),
        return_messages=True,
        memory_key="chat_history",
        ouptut_key="answer",
        k=2,
    )
