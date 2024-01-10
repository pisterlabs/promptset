import os

import boto3

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


from Application.memory import Memory

OPENAI_API_key=os.getenv('OPENAI_API_key')
llm = ChatOpenAI(openai_api_key = OPENAI_API_key)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("you are an assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_message}"),
    ] 
)

class TestMemory():
    
    def test_creation(self):
        memory = Memory(memory_type="DynamoDB")
        assert memory.memory_type == "DynamoDB"
        assert memory.SessionTable_name == "SessionTable"
        
        dynamodb = boto3.resource("dynamodb")
        
        assert memory.SessionTable_name in [table.name for table in dynamodb.tables.all()]
        
        
        
    def test_save_and_load_data(self):
        memory = Memory(memory_type="DynamoDB")
        messages = ["hello","how are you"]
        conversation_memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True, max_history_length=20)
        chatbot = LLMChain(llm=llm, prompt=prompt, memory = conversation_memory, verbose=False)
        for message in messages:
            chatbot.memory.chat_memory.add_ai_message(message)
        memory.save_history_in_dynamodb(chatbot, chat_id="1")
        chat_memory = memory.load_history_from_dynamodb(chat_id="1")[0]
        conversation_memory1 = ConversationBufferMemory(memory_key="chat_history",return_messages=True,chat_memory=chat_memory, max_history_length=20)
        chatbot1 = LLMChain(llm=llm, prompt=prompt, memory = conversation_memory1, verbose=False)
        
        assert chatbot1.memory.chat_memory.messages[0].content == "hello"
        assert chatbot1.memory.chat_memory.messages[1].content == "how are you"
        
        
        
    