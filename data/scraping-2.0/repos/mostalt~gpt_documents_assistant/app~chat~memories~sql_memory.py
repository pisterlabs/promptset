from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from pydantic import BaseModel

from app.web.api import (
  get_messages_by_conversation_id,
  add_message_to_conversation
)
from app.chat.memories.histories.sql_history import SqlMessageHistory


def build_memory(chat_args):
  return ConversationBufferMemory(
    chat_memory=SqlMessageHistory(conversation_id=chat_args.conversation_id),
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
  )