from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd


df = pd.read_csv('data/titanic.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

agent.run("how many rows are there?")


#llm = OpenAI(api_token = "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")
