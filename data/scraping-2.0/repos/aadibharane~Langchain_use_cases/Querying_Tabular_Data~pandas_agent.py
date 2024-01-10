# #Pandas Dataframe Agent
'''
This notebook shows how to use agents to interact with a pandas dataframe. It is mostly optimized for question answering.

NOTE: this agent calls the Python agent under the hood, which executes LLM generated Python code - this can be bad if the
LLM generated Python code is harmful. Use cautiously.
'''
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

def pandas_agent():
    df = pd.read_csv('titanic.csv')

    # Single DataFrame example
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    agent.run("how many rows are there?")
    agent.run("how many people have more than 3 siblings?")
    agent.run("what's the square root of the average age?")
pandas_agent()
# Multi DataFrame example
# df1 = df.copy()
# df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
# merged_df = pd.concat([df, df1], axis=1)  # Concatenate the DataFrames horizontally

# agent = create_pandas_dataframe_agent(OpenAI(temperature=0), merged_df, verbose=True)
# agent.run("how many rows in the age column are different?")
