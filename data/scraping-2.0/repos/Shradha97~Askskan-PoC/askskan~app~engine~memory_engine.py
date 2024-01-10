import json
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict


class MemoryEngine:
    @classmethod
    def get_chat_db_memory(cls):
        pass

    @classmethod
    def get_buffer_memory(
        cls, chat_history=ChatMessageHistory()
    ) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            memory_key="chat_history", input_key="question", chat_memory=chat_history
        )

    @classmethod
    def get_chat_history(
        cls,
        buffer_memory: ConversationBufferMemory,
    ):
        chat_history = buffer_memory.chat_memory

        # FIXME: replace this with retrieving from a db

        # retrieve_from_db = json.loads(json.dumps(chat_messages_dict))
        # retrieved_messages = messages_from_dict(retrieve_from_db)

        return chat_history
