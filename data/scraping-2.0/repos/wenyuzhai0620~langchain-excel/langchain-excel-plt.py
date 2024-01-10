import os
import re
import matplotlib.pyplot as plt
import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
import tempfile

# your apikey

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


# creating app
st.title('🦜🔗 GPT Excel Helper')
st.write('Please upload your excel file and write in your prompt to get started')


llm = AzureChatOpenAI(temperature=0, deployment_name='gpt-35-turbo',model='gpt-35-turbo')

def csv_agent_func(file_path, user_message):
    agent = create_csv_agent(
        llm,
        file_path, 
        verbose=True,
    )

    try:
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        print('tool_input', tool_input)
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None


uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)
    print("tmp_file_path", uploaded_file.name)
    file_path = os.path.join("WDI_CSV/", uploaded_file.name)
   
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(file_path)
    st.dataframe(df)
    user_input = st.text_input("Enter your query here")
    if st.button('Run'):
        response = csv_agent_func(file_path, user_input)
        print('user_input', user_input)
        print('file_path', file_path)
        print('response', response)

        st.write(response)

    st.divider()