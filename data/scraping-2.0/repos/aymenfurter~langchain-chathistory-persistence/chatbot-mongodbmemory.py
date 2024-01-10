from langchain.memory import ConversationBufferMemory 
import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import MongoDBChatMessageHistory


api_key = os.environ.get('OPEN_AI_KEY')
connection_string = os.environ.get('MONGODB_CONNECTION_STRING')

user = input("Please input your username: ")

message_history = MongoDBChatMessageHistory(
    connection_string=connection_string, session_id=user
)

memory = ConversationBufferMemory(
    memory_key="history", chat_memory=message_history, return_messages=True
)


chain = ConversationChain(
    llm=OpenAI(temperature=0), 
    memory=memory, 
    verbose=True
)

history = []

while True:
    query = input("Enter Your Query:")
    print(chain.predict(input=query))