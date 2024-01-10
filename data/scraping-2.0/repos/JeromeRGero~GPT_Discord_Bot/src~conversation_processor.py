from discord.message import Message
from discord.threads import Thread
import openai
import langchain
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.mongodb import MongoDBChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    messages_from_dict, 
    messages_to_dict
)
from config import *
from helper import *

async def process_conversation(user_message, id_conversation_name, message):
    # ------------------ OpenAI ------------------
    
    # system_message_content = "You are a helpful assistant that translates English to French."

    # system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_content)  

    # prompt = ChatPromptTemplate.from_messages([
    #     system_message_prompt,
    #     MessagesPlaceholder(variable_name="history"),
    #     HumanMessagePromptTemplate.from_template("{input}")
    # ])
    

    history: MongoDBChatMessageHistory = MongoDBChatMessageHistory(
        mongo_db_url, id_conversation_name, mongo_db, conversations_collection)

    memory: ConversationBufferMemory = ConversationBufferMemory(return_messages=True, chat_memory=history)

    llm = OpenAI(model_name="gpt-4", temperature=0.4)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory,
        # prompt=prompt
    )

    response: str = conversation.predict(input=user_message)
    print("response: ", response)
    
    # Send the response in parts no longer than 2000 characters.
    if len(response) > 2000:
        x=0
        while x < len(response):
            await message.reply(response[x:x+2000])
            x+=2000
    else:
        await message.reply(response)