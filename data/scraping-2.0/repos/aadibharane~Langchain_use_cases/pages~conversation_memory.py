import os
import streamlit as st
import langchain.memory
import langchain.llms
import langchain.chains
from apikey import apikey
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] =apikey 

def conversation_memory():
    history = ChatMessageHistory()
    history.add_user_message("hi!")
    history.add_ai_message("whats up?")

    memory = ConversationBufferMemory(chat_memory=history)

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

    def predict(user_input):
        response = conversation.predict(input=user_input)
        return response

    st.title("Conversation Memory Chatbot")

    user_input = st.text_input("Enter your message:")

    if user_input:
        response = predict(user_input)
        st.write("AI response:", response)


if __name__ == "__main__":
    conversation_memory()  
