import os
from langchain.agents import *
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import streamlit as st
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

db_user = "postgres"
db_password = "Pass1234"
db_host = "database-1.cojmud51692g.us-east-1.rds.amazonaws.com"
db_name = "postgres"
db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
#db = SQLDatabase.from_uri(f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}",)

QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

# Setup the database chain
def generate_response(txt,openai_api_key):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    #db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    ques=QUERY.format(question=txt)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
    # Split text    
    return db_chain.run(ques)

# Page title
def generate_response(txt,openai_api_key):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    #db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    ques=QUERY.format(question=txt)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
    # Split text    
    return db_chain.run(ques)

# Page title
st.set_page_config(page_title='Chat with your DB')
st.title('Chat with your DB')

tab1, tab2= st.tabs(["Chat with Pretrained DB", "Train"])
# Text input

with st.sidebar:
    st.button("Chat with DB")
    with tab1:
        openai_api_key = st.text_input('OpenAI API Key', type='password')
        txt_input = st.text_area('Enter your text', '', height=200)
        # Form to accept user's text input for summarization
        result = []
        with st.form('chatwithDB_form', clear_on_submit=True):
            #openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
            submitted = st.form_submit_button('Submit')
            if submitted:
                with st.spinner('Calculating...'):
                    response = generate_response(txt_input,openai_api_key)
                    result.append(response)
        if len(result):
            st.info(response)    
    with tab2:
        username = st.text_input('Enter username')
        password = st.text_input('Enter password',type = 'password')
        host = st.text_input('Enter Hostname')
        port = st.text_input('Enter Port')
        dbname = st.text_input('Enter DBName')
        with st.form('TrainDB_form', clear_on_submit=True):
            train = st.form_submit_button('Train')

with st.sidebar:       
    st.button("Q&A on Docs")   
