import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("CodeGenius")

if "cgmessages" not in st.session_state:
    st.session_state["cgmessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.cgmessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.cgmessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""Welcome to CodeGenius, your AI software developer companion. As a seasoned developer with {Years_of_Experience} years of hands-on coding and a proven track record at companies like {Past_Companies}, I am here to assist you in all aspects of software development. Leveraging my expertise in {Industry_Experience}, I can help you with coding challenges, provide guidance on software architecture and design patterns, and recommend tools and frameworks that align with your project requirements. How can I help you in your coding journey today?"""),
                HumanMessage(content="You: "+message)]
    response = chat(messages)
    st.session_state.cgmessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
