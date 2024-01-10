import streamlit as st
import json
import argparse
import os
import pandas as pd
from typing import Any, Dict
# Additional import for SQL connection
from sql_connector import SqlConnector
from langchain.llms import OpenAI
from collections import namedtuple

SqlConnector = namedtuple('SqlConnector', 'query')

# Load json file
def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_txt(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = file.read()
    return data

# Save json file
def save_json(file_path: str, data: Dict[str, Any]):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# Get final path
def get_final_path(path_type: int, paths: list) -> str:
    current_directory = os.getcwd()
    final_path = os.path.join(current_directory, *paths)
    return final_path


class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

class ChatSql:

    def __init__(self) -> None:
        conf_path: str = get_final_path(1, ["conf.json"])
        self.conf: Dict[str, Any] = read_json(conf_path)
        self.llm: object = OpenAI(openai_api_key=self.conf["OPEN_AI_KEY"])
        self.info: str = str(read_json(get_final_path(1, ["info.json"])))
        self.data: str = str(read_txt(get_final_path(1, ["data.txt"])))

    def prompt_to_query(self, prompt: str) -> Dict[str, str]:
        info = self.info
        data= self.data
        template = """
        Your mission is convert SQL query from given {prompt}. Use following database information for this purpose (info key is a database column name and info value is explanation) : {info} .  along with this i am sharing some sample data from this table :  {data}
        --------
        Put your query in the  JSON structure with key name is 'query'
        """
        pr_ = PromptTemplate(input_variables=["prompt", "info"], template=template)
        final_prompt = pr_.format(
            prompt=prompt,
            info=info,
            data=data
        )
        gpt_query: Dict["str", "str"] = json.loads(self.llm(final_prompt))

        return gpt_query

    def query_to_result(self, gpt_query: Dict[str, str]) -> str:
        raw_res: str = SqlConnector(query=gpt_query["query"]).query
        return raw_res

    def raw_result_to_processed(self, raw_result: str) -> str:
        res_processing_template = """
        Your mission is convert database result to meaningful sentences. Here is the database result: {database_result}. validate the query before responding. 
        """
        db_pr = PromptTemplate(
            input_variables=["database_result"], template=res_processing_template
        )
        final_prompt = db_pr.format(database_result=raw_result)
        procesed_result: str = self.llm(final_prompt)

        return procesed_result

csql = ChatSql()

st.title('ChatSql Application')
prompt = st.text_input("Enter your SQL prompt")
api_key = st.text_input("Enter your OpenAI API Key", type="password")

conf_path = get_final_path(1, ["conf.json"])
info_path = get_final_path(1, ["info.json"])

# Load the contents of conf.json and info.json
conf_content = read_json(conf_path)
info_content = read_json(info_path)

# On button press, start the operation
if st.button('Submit'):
    with st.spinner('Processing...'):
        csql.llm = OpenAI(openai_api_key=api_key)  # update the API Key
        query = csql.prompt_to_query(prompt)
        st.write("Generated SQL Query:")
        st.write(query["query"])
        #result = csql.query_to_result(query)
        #processed_result = csql.raw_result_to_processed(result)
        #st.write("Processed Result:")
        #st.write(processed_result)
