# Standard Library Imports
import json
import os
import re
import time
import warnings

# Third-Party Library Imports
import numpy as np
import pandas as pd
import streamlit as st

# Langchain Library Imports
from langchain import (
    LLMChain,
    PromptTemplate,
    OpenAI,
)
from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
    create_csv_agent,
    create_pandas_dataframe_agent,
    load_tools,
    ZeroShotAgent,
    AgentExecutor,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


############################################################################################################
# Load environment variables

# HOSTED
API_KEY = None
# API_KEY = st.secrets["apikey"]

############################################################################################################

# Initialize session state variables
if "headings_list" not in st.session_state:
    st.session_state.headings_list = ""

# 3. Load Data Function
def load_data(path):
    '''This function loads a csv file from the provided path and returns a pandas DataFrame'''
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_headings(df):
    headings = df.columns.tolist()
    return headings
        
    # Return the list of headings
    return headings

# Function to preview the column headings of the uploaded CSV file
def show_headings():
    if st.session_state.data is not None:
        headings = get_headings(st.session_state.data)
        st.session_state.headings_list = "\n".join(headings)


# Function to process the DataFrame and generate insights
def df_agent(df, agent_context, describe_dataset, query):
    if API_KEY is None:
        st.error("Please enter the password or your API key to proceed.")
        return
    llm = OpenAI(openai_api_key=API_KEY,temperature=0) 
    # llm = ChatOpenAI(openai_api_key=API_KEY,temperature=0, model_name='gpt-4') <- Trial with ChatGPT 4
    df_agent_research = create_pandas_dataframe_agent(llm, df, handle_parsing_errors=True)
    df_agent_analysis = df_agent_research(
        {
            "input": f"You are DataFrameAI, the most advanced dataframe analysis agent on the planet. You are collaborating with a company to provide skilled, in-depth data analysis on a large table. They are looking to gain competitive business insights from this data, in order to gain an edge over their competitors. They are looking to analyze trends, ratios, hidden insights, and more. \
                You are a professional data science and analysis agent with the following strengths: {agent_context} \
                The dataset can be described as follows: {describe_dataset} \
                Specifically, they are looking to answer the following question about the data: {query} \
                OUTPUT: Provide detailed, actionable insights. I am not looking for one or two sentences. I want a paragraph at least, including statistics, totals, etc. Be very specific, and analyze multiple columns or rows against each other. Whatever is required to provide the most advanced information possible!"
        }
    )
    df_agent_findings = df_agent_analysis["output"]
    return df_agent_findings


############################################################################################################

# STREAMLIT APP
st.title("👨‍💻 Query your CSV with an AI Agent using Langchain")
st.write("Beyond a basic CSV Agent to query your tabular data, this app allows you to provide a prompt to the agent, preview headings, provide task objectives, and contextual information about your data.")
st.write("Uses OpenAI. You need the key, or...hit me up if you're cool and I can give you the password!")

# Add a password input
password = st.text_input("Enter the password to use the default API key")

# Check if the password is correct
if password == st.secrets["password"]:
    API_KEY = st.secrets["apikey"]
else:
    API_KEY = st.text_input("Enter your own API key", type='password')

uploaded_file = st.file_uploader("Please upload your CSV file below")


if uploaded_file is not None:
    if uploaded_file.size == 0:
        st.write("The uploaded file is empty.")
    else:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.write("The uploaded file does not contain any data or columns.")
else:
    st.session_state.data = None

if st.button("PREVIEW HEADINGS", type="secondary", help="Click to preview headings", on_click=show_headings):
    pass

# Display the headings text area
headings_list = st.text_area(label="Headings", value=st.session_state.headings_list, key="headings")

describe_dataset = st.text_area("Please describe your dataset. e.g., 'This is Amazon sales data that contains XYZ.'")
objectives = st.text_area("Describe your objectives. e.g., 'I am specifically looking for data insights related to overlooked ratios, key performance indicators, or hidden insights. Test correlations or complete data analysis when required.'")
agent_context = st.text_area("Agent context prompt. e.g., 'You are a skilled data scientist. You are looking for trends, ratios, and actionable insights into the data. Your answers will result in marketing spend decisions, so be as specific as possible.'")
query = st.text_area("Type your query")


if st.session_state.data is not None:
    if isinstance(st.session_state.data, pd.DataFrame):
        if st.button("Submit Query"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            max_time = 25  # You should estimate how long the task will take

            for i in range(max_time):
                progress_bar.progress((i + 1) / max_time)
                status_text.text(f'Analyzing Data: {i+1}')
                time.sleep(1)  # Sleep for a second to slow down the progress

            status_text.text('Running query...')
            dataframe_insights = df_agent(st.session_state.data, agent_context, describe_dataset, query)
            progress_bar.empty()  # You can empty the progress bar here

            # status_text.text('Query Completed')  # Updating the status text
            # st.markdown(f'<div style="font-size: 18px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2); padding: 10px">{dataframe_insights}</div>', unsafe_allow_html=True)
            status_text.text('Query Completed')  # Updating the status text

            markdown_style = '''
                <style>
                .custom-markdown {
                    background-color: #f2edfe;
                    color: #000000;
                    padding: 10px;
                    box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
                    font-size: 19px;
                }
                </style>
            '''

            markdown_html = f'<div class="custom-markdown">{dataframe_insights}</div>'
            st.markdown(markdown_style, unsafe_allow_html=True)
            st.markdown(markdown_html, unsafe_allow_html=True)

