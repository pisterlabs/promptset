import pandas as pd
from langchain.memory import ConversationBufferWindowMemory
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from settings import FILE_PATH
from llm import llm

memory = ConversationBufferWindowMemory(memory_key="chat_history",return_messages=True,k=5)

def create_csv_agent(llm, path):
  PREFIX = """
  You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
  You should use the tools below to answer the question posed of you:

  Chat history:
  {chat_history}

  Note: If the given question does not contain relavant information to answer the question, come up with questions that can be used to answer and return as response."
  """
  df = pd.read_csv(path)
  pd_agent = create_pandas_dataframe_agent(
      llm,
      df,
      prefix=PREFIX,
      verbose=True,
      agent_executor_kwargs={"memory": memory},
      input_variables=['df_head', 'input', 'agent_scratchpad', 'chat_history']
  )
  return pd_agent

def recommend_houses_agent(llm, path):
  PREFIX = """
  You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
  You should use the tools below to answer the question posed of you:

  Chat history:
  {chat_history}

  You are to recommend certain house based on the relavant information provided e.g Recommend based on price, location, number of bedroom, etc.
  Note: If the given input does not contain relavant information to answer the question, come up with questions that can be used to provide awesome recommendation of houses to buy."
  """
  df = pd.read_csv(path)
  pd_agent = create_pandas_dataframe_agent(
      llm,
      df,
      prefix=PREFIX,
      verbose=True,
      agent_executor_kwargs={"memory": memory},
      input_variables=['df_head', 'input', 'agent_scratchpad', 'chat_history']
  )
  return pd_agent


csv_agent = create_csv_agent(
    llm,
    FILE_PATH
)
house_buy_agent = recommend_houses_agent(
    llm,
    FILE_PATH
)