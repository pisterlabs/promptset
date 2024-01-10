import streamlit as st 
import stock_tools_lib as glib 
import stock_tools_database_lib as databaselib 
from langchain.callbacks import StreamlitCallbackHandler
import time
import pandas as pd

def print_result(st, response):
    st.write(response['output'])

def stock_tools():
    st.header("Stock Tools Agent")
    st.write("Try: Company name as Amazon, Tesla, Apple..etc ")
    st.write("get company ticker of Amazon")
    st.write("get stock history of Tesla")
    st.write("get financial statement of Vinamilk")
    st.write("fetch news of Apple")

    if 'database' not in st.session_state: 
        with st.spinner("Initial Database"): 
            databaselib.initial_database() 
        
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = [] 

    agent = glib.initializeAgent()
    input_text = st.chat_input("Type company name here!") 
    ph = st.empty()
    if input_text:
        ph.empty()
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent({
            "input": "\n\nHuman:" + str(input_text) + " \n\nAssistant:",
            "chat_history": st.session_state.chat_history,
            "output":"output"
         },
            callbacks=[st_callback])
        st.write(response['output'])


    

