# from langchain.schema.messages import BaseMessage
# from pydantic import BaseModel 
from langchain.memory import ConversationBufferMemory
# from langchain.schema import BaseChatMessageHistory

from app.chat.memories.histories.sql_history import SqlMessageHistory

# from app.web.api import (
#     get_messages_by_conversation_id, 
#     add_message_to_conversation
# )


# # class to extend BaseChatMessageHistory & pydantic BaseModel
# class SqlMessageHistory(BaseChatMessageHistory, BaseModel):
#     # need to know convo working with have conversation_id attribute
#     conversation_id: str

#     @property
#     def messages(self): 
#         return get_messages_by_conversation_id(self.conversation_id)
    
#     def add_message(self, message):
#         return add_message_to_conversation(
#             conversation_id=self.conversation_id, 
#             role=message.type, 
#             content=message.content
#         )

#     def clear(self): 
#         pass 


# called with a chat_args object 
def build_memory(chat_args): 
    '''make a ConversationBufferMemory & override the default mess chat_memory
    property with an instance of SqlMessageHistory'''
    return ConversationBufferMemory(
        chat_memory=SqlMessageHistory(
            conversation_id=chat_args.conversation_id
        ),
        return_messages=True, 
        memory_key="chat_history", 
        output_key="answer"
    )

