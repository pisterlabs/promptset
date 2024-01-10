
from data_processor import DataProcessor
from meta_data_service import MetaDataService
from data_pipeline_executor import DataPipelineExecutor
from pandas import DataFrame
import pandas as pd
import openai
import llm_wrapper as llmwrapper
import os
from dotenv import load_dotenv
import completion_builder as cb
from user_session_state import UserSessionState
from actions import ActionsManager


load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")

meta_data_service = MetaDataService()
meta_data_service.add_data_source("backend/datasources/nicktrialbalance.json", "backend/datasources/nicktrialbalance.csv")
action_manager = ActionsManager("gpt-4-1106-preview", meta_data_service)
user_session_state = UserSessionState()


test_input = "load balances, then filter it on company code 0302 - call the output 0302_balances"

data,metadata,commentary = action_manager.function_generate_pipeline_definition(None,None,user_session_state, test_input)
data2, metadata2, commentary2 = action_manager.execute_pipeline_definition(None,None,user_session_state,"create that","MyNewDataSet", "Data set description")

print(metadata)
print(commentary2)
