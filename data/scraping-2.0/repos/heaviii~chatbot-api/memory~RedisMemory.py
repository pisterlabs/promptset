import json
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from config import Config
from langchain import OpenAI
from langchain.schema import (
    BaseChatMessageHistory,
    BaseMessage,
    _message_to_dict,
    messages_from_dict,
)
from typing import List, Optional


class RedisMemory(RedisChatMessageHistory):

    def __init__(self, session_id: str):
        super().__init__(session_id, url=Config.REDIS_URL, key_prefix="chat_history", ttl=Config.REDIS_TTL)
    

    def get_memory_db(self, session_id):
        message_history = self#.__init__(self,url=Config.REDIS_URL, ttl=Config.REDIS_TTL, session_id=session_id, key_prefix="chat_history")
        #message_history.clear()
        return message_history
    
    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Redis"""
        _items = self.redis_client.lrange(self.key, 0, limit)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        messages = messages_from_dict(items)
        return messages
    

    def get_summary_memory(self):
        message_history = self
        #print("message_history:---",(message_history.messages))
        summary_memory = ConversationSummaryMemory(llm=OpenAI(verbose=True), input_key="input", chat_memory = message_history)
        return summary_memory

    def summary_memory(self, session_id):
        # Combined
        message_history = self

        conv_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            chat_memory = message_history,
        )
        summary_memory = ConversationSummaryMemory(llm=OpenAI(verbose=True), input_key="input", chat_memory = message_history)

        #print("conv_memory:---",summary_memory.chat_memory.messages)
        memory = CombinedMemory(memories=[conv_memory, summary_memory])
        return memory
    