import json
import openai
import os
import pandas as pd

from os import getenv
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent, initialize_agent, load_tools, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI

load_dotenv()

# environment initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# variable initialization
OPENAI_API_KEY = getenv("OPENAI_KEY")
openai_api_key = OPENAI_API_KEY

def agent_answer(df, natural_language_query):

    prompt = (
        f"""
        You are an expert data analyst whose goal is to give an answer, as accurate as possible, to a user's question. You are given a question from a user and a pandas dataframe that contains the data to answer this question.\n
        The question is the following:
        ---------------\n
        {natural_language_query}
        \n---------------\n
        Explain your reasoning to reach the answer.
        Take into consideration the data type and formatting of the columns.
        Its possible that any product/service or other variables the user is referring to appears with a different name in the dataframe. Explain this in your answer in a polite manner, but always trying to give an answer with the available data.
        Lets think it through step by step.
        """
    )

    llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0, openai_api_key=openai_api_key)
    agent =  create_pandas_dataframe_agent(llm, df, verbose=True)
    response = agent.run(prompt)
    print("\nAgent answer:", response)
    print("\n", type(response))
    return response