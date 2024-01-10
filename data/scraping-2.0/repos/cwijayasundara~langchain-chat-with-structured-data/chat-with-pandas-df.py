import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm=ChatOpenAI(temperature=0)

df = pd.read_csv('test.csv', sep=';')
print(df.head())

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True)

st.title("Chat With Your Pandas DF ! - Natural Language to DF Ops ..")
st.text_input("Please Enter Your Query in Plain Text ! ", key="query")
result = agent.run(st.session_state.query)
st.write(result)
