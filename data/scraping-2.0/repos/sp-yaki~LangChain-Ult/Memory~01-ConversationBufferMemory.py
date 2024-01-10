from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hello, nice to meet you!")
conversation.predict(input="Tell me about the Einstein-Szilard Letter ")
print(memory.buffer)
memory.save_context({"input": "Very Interesting."}, 
                    {"output": "Yes, it was my pleasure as an AI to answer."})
print(memory.load_memory_variables({}))

import pickle
pickled_str = pickle.dumps(conversation.memory)

with open('memory.pkl','wb') as f:
    f.write(pickled_str)
    
new_memory_load = open('memory.pkl','rb').read()

llm = ChatOpenAI(temperature=0.0)
reload_conversation = ConversationChain(
    llm=llm, 
    memory = pickle.loads(new_memory_load),
    verbose=True
)

print(reload_conversation.memory.buffer)