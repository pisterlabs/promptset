## Conversational Q&A Chatbot

import streamlit as st
from dotenv import load_dotenv
import os
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title='Conversational QnA Chatbot Demo', page_icon=':books:')
st.title("Hey! Let's chat!")

load_dotenv()

chat_llm = ChatOpenAI(temperature=0.6, openai_api_key=os.getenv('OPENAI_API_KEY'))

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content= 'You are a conversational AI assistant')
    ]

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content = question))
    answer = chat_llm(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content = answer.content))
    
    return answer.content

input_question = st.text_input('Question: ', key='question')
response = get_chatmodel_response(input_question)
submit = st.button('Ask')

if submit:
    st.subheader('Response')
    st.write(response)