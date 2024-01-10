from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from streamlit_chat import message
import streamlit as st
import pandas as pd
import openai
from PIL import Image
import os
# from dotenv import load_dotenv, find_dotenv

st.set_page_config(layout="wide",page_title="ExcelMate",page_icon="https://cdn.dribbble.com/userupload/3963238/file/original-2aac66a107bee155217987299aac9af7.png?compress=1&resize=400x300&vertical=center")
image = Image.open(r"analysis.webp")
st.sidebar.title("Xccelrate")
st.sidebar.text("Accelrate your work 🏎️")
st.sidebar.image(image)


st.set_option('deprecation.showPyplotGlobalUse', False)

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")

def pandas_agent(df,user_prompt):
    agent = create_pandas_dataframe_agent(OpenAI(engine="gpt-demo",temperature=0), df, verbose=True)
    return agent.run(user_prompt)

def desc(df):
    agent = create_pandas_dataframe_agent(OpenAI(engine="gpt-demo",temperature=0), df, verbose=True)
    return agent.run("Describe the data and provide some insights of the data in tabular format")

# st.sidebar.title("ExcelMate")
# st.sidebar.caption("Your go-to solution for all Excel queries. Get instant answers and solutions for your Excel file questions.")
excel_file = st.sidebar.file_uploader("Upload",type="csv")
# user_prompt = st.text_input("",placeholder="Ask Your Query..")

if excel_file is not None:
    df = pd.read_csv(excel_file)
    st.sidebar.dataframe(df)

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
            
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + excel_file.name + " 🤗"] 
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
                
            user_input = st.text_input("Query:", placeholder="Interact with your data here...", key='input')
            submit_button = st.form_submit_button(label='Send')

        st.session_state['past'].append("Describe the data.")
        output = desc(df)
        st.session_state['generated'].append(output)

        if submit_button and user_input:
            output = pandas_agent(df,user_input)
            st.pyplot()
                
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
