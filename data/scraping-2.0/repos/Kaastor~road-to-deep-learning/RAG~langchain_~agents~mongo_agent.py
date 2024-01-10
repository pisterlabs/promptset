import ast
import os
import re

import openai
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

from ast import literal_eval
from dateutil.parser import parse

from mongo_connector import MongoConnector


def convert_string_to_dict(input_str):
    # Replace MongoDB-specific data types with Python-compatible strings
    input_str = re.sub(r'ISODate\("(\d{4}-\d{2}-\d{2})"\)', r'"\1"', input_str)

    # Use literal_eval to convert the sanitized string to a dictionary
    parsed_dict = literal_eval(input_str)

    # Recursively convert date strings to datetime objects
    def convert_dates(d):
        for key in d.keys():
            if isinstance(d[key], dict):
                convert_dates(d[key])
            else:
                if re.match(r'\d{4}-\d{2}-\d{2}', str(d[key])):
                    d[key] = parse(d[key])

    convert_dates(parsed_dict)
    return parsed_dict


'''
1. Create a prompt to get data from mongo.
2. Get data from mongo
3. Use pandas agent to analyse the data

Questions:
- Can agent be reused? In one-shot questions
- How to make agent safe? Run on Pod with some environmental restrictions?
'''

openai.api_key = os.environ["OPENAI_API_KEY"]

# 1. Prompt to get subset of data
q_template = (
    "You are a helpful assistant. You should act as an expert in creating pymongo queries and focus on that task.\
    You will answer questions related to mongodb collection with following fields \
    (fields are presented in the following format: <field name>:<field mongo type>:<description>): \
    ```\
    _id:String:unique identifier of document \
    contributors:Array:email addresses of users which can read the file\
    readers:Array:email addresses of users which can read the file\
    created:Date:date when file was created\
    fileSize:Int64:size of the file in bytes\
    mimeType:String:mime type of file\
    owner:String:email address of user which owns the file\
    updated:Date:date when file was changed in any way\
    title:String:title of file\
    ```\
    Use these fields to construct filter for mongodb query in python. Constructed filters should always \
    have ONLY string values (no python function calls).\
    Your answer should be ONLY the query which can be applied to `filter` argument in pymongo `find` method.\
    In other words you should only return python dict, NO ADDITIONAL TEXT.\
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.\
    Instruction from user: {instruction}"
)
prompt = PromptTemplate.from_template(q_template)

# 2. Get the query
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
chain = LLMChain(llm=llm, prompt=prompt)

# instruction = "Return me files of user `susan@generalaudittool.com` created before 2023-07-07"
instruction = "Return me all files"

with get_openai_callback() as cb:
    response_query = chain.run(instruction=instruction)
    print("Query: ", response_query)

    # 2. Get the data
    connector = MongoConnector(database='C03i2p9bk-drive')
    query = convert_string_to_dict(response_query)
    data = connector.query(query, collection='file_meta')
    data_df = pd.DataFrame(list(data))

    # 3. Use Agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=data_df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    instruction = "Calculate average file size for each type of file."

    # agent.run("Which file has biggest size? Return it's title.")
    # agent.run("Calculate average file size.")
    # agent.run("Calculate average file size. Tell result in MB.")
    # $0.005735 (37 records)
    # $0.008469 (178255 records)
    # response = agent.run("Calculate average file size for each type of file.")
    # response = agent.run("Which user has the most files?")
    response = agent.run("Give top 5 users that have in total the largest files? Give top 10 users that have in total the largest files? Give top 15 users that have in total the largest files?")


    print("Total cost: $", cb.total_cost)
