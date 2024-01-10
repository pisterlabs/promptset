import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferWindowMemory(k=1)

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hello, nice to meet you!")

# print(memory.buffer)

conversation.predict(input="What day is it today?")

# memory.save_context({"input": "Very Interesting."}, 
#                     {"output": "Yes, it was my pleasure as an AI to answer."})

print(memory.load_memory_variables({}))