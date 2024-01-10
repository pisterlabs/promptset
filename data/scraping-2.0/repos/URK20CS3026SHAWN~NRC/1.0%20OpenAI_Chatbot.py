# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

# Constants
COHERE_API_KEY = apikey

#Chat Class
class Chat:
    def __init__(self, llm=None, memory=None, prompt=None):
        self.llm = llm
        self.memory = memory
        self.prompt = prompt
        self.chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='title', memory=memory)
    
    def run(self, prompt):
        return self.chain.run(prompt)

    def predict(self, prompt):
        return self.chain.predict(prompt)

    def reset(self):
        self.memory.reset()
        self.chain.reset()

    def read_memory(self):
        return self.memory.buffer


# App framework
st.title('🦜🔗 Langchain With OPENAI')


Chats = []

prompt = st.text_input('Tell Sakhi what you want to talk about') 

# Memory 
memory = ConversationBufferMemory(input_key='topic',ai_prefix="AI_Mem",human_prefix="You", memory_key='chat_history')

# Llm
llm = Cohere(cohere_api_key=COHERE_API_KEY, temperature=0, max_tokens=250) 
# First Chat
Chats.append(Chat(llm = llm, memory = memory, prompt = PromptTemplate(input_variables = ['topic'], template='Write about the {topic}')))


st.write("MEM: "+Chats[0].read_memory()+"\n")
Chats[0].run(prompt)
st.write("AFM: "+Chats[0].read_memory())




