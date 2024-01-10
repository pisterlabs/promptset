from langchain.memory import ConversationBufferMemory

from app.chat.memory.history.sql_history import SqlMessageHistory


def build_memory(chat_args):
    return ConversationBufferMemory(
        chat_memory=SqlMessageHistory(conversation_id=chat_args.conversation_id),
        return_messages=True,
        memory_key="chat_history",
        ouptut_key="answer",
    )
