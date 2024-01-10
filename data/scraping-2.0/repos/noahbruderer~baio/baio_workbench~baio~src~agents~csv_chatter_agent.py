from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from src.llm import LLM
llm = LLM.get_instance() 


def csv_agent_creator(path: str):
    """Input: path to csv, will be loaded as panda and input in pandas_dataframe_agent when it is initiated
        
       RETURN: agent 
    """
    
    df1 = pd.read_csv(path[0])
    print(path)
    if path and len(path) > 1 and path[1] != '':
        df2 = pd.read_csv(path[1])
        csv_chatter_agent = create_pandas_dataframe_agent(llm, 
                                                [df1,df2], 
                                                verbose=True, 
                                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                                )
        return csv_chatter_agent        
    else: 
        csv_chatter_agent = create_pandas_dataframe_agent(llm, 
                                                df = df1, 
                                                verbose=True, 
                                                agent_type=AgentType.OPENAI_FUNCTIONS, 
                                                )
        return csv_chatter_agent
