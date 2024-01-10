import os
from langchain.llms import OpenAI
from config.settings import settings
from langchain.agents import load_tools
from langchain.agents import initialize_agent

os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["SERPAPI_API_KEY"] = settings.serpapi_api_key


# llm = OpenAI(temperature=0.9)  # high creativity
llm = OpenAI(temperature=0)  # no creativity
tools = load_tools(["serpapi"], llm=llm)

# initialize an agent with the tools
agent = initialize_agent(tools, llm, verbose=True)
prompt = "How many airplanes are in the air right now?"
agent.run(prompt)


# import pandas as pd
#
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import create_pandas_dataframe_agent
#
# df = pd.read_csv("orderdata.csv")
# chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
# agent = create_pandas_dataframe_agent(chat, df, verbose=True)
# agent.run("what is the total revenue generated from all orders")
