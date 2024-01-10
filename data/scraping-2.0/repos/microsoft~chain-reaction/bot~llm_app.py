import os
import openai
import pandas as pd
import numpy as np
import json
from typing import List, Optional
import requests
from dotenv import dotenv_values
import psycopg2
from psycopg2 import pool
from psycopg2 import Error
from psycopg2 import sql
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import glob
import pdb

# Find file name in same folder as this file that ends with *.env
current_dir = os.path.dirname(os.path.abspath(__file__))  
env_files = [file for file in glob.glob(os.path.join(current_dir, '*.env')) if "example.env" not in file]  

# Get the absolute path to the .env file in the current folder
try:
    env_name = os.path.join(os.path.dirname(__file__), env_files[0])
except:
    folder = os.path.basename(os.getcwd())
    raise Exception("No *.env file found in /{}. Please create one. If you've added your keys to example.env, please use a different file name because we want to prevent accidental commits of keys in git".format(folder))

# Load environment variables from the .env file
config = dotenv_values(env_name)

for key, value in config.items():
    os.environ[key] = value

# LOAD OpenAI configs
openai.api_type = config["OPENAI_API_TYPE"]
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = config['OPENAI_API_VERSION']
print("ENV VARIABLES LOADED")

def get_prompt_template(prompt_id, prompt_templates_name=None):
    """
    Retrieve LLM prompt template using prompt_id from a csv file.
    """
    if prompt_templates_name == None:
        prompt_templates_name = config['PROMPT_TEMPLATE_FILE']
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), prompt_templates_name))
    prompt = df[df['prompt_id'] == prompt_id]['prompt_template'].values[0]
    return prompt

def llm_call(msg, prompt_template, context=None):
    """
    Use llm with prompt template to generate the answer. If there is context
    in the prompt template, be sure to include it.
    """
    engine = "gpt-35-turbo"
    llm = AzureChatOpenAI(
        deployment_name=engine,
        openai_api_base=openai.api_base,
        openai_api_version=openai.api_version,
        openai_api_key=openai.api_key,
        openai_api_type = openai.api_type,
        temperature=0.0,
    )
    
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(prompt_template),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=template)

    if context == None:
        ans = chain.run(
            {
                'text': msg,
            }
        )
    else:
        ans = chain.run(
            {
                'context': context,
                'text': msg,
            }
        )
    print(ans)
    return ans

def test_two_prompt(msg):
    # Sequence of four functions to generate answer using two LLM calls
    prompt_id_a = 0
    prompt_template_1 = get_prompt_template(prompt_id_a)
    intermediate_ans = llm_call(msg, prompt_template_1, context=None)

    prompt_id_b = 1
    prompt_template_2 = get_prompt_template(prompt_id_b)
    ans = llm_call(msg, prompt_template_2, context=intermediate_ans)
    return ans

if __name__ == '__main__':
    question = "A box has a small hole in it with 2 light bulbs that can come on/off independently. Either bulb can be on 80% of the time. What % of time can both bulbs both be on or off?"
    context = "Bayes Theorem"
    prompt_template = """Use the following information to answer the question below. 
    Theory: {context} 
    Respond as if you are trying to explain your thought process in a job interview setting.
    """

    ans = llm_call(question, prompt_template, context)
    print(ans)
