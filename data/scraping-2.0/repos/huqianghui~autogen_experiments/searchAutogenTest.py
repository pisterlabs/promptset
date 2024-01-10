import os
from autogen import config_list_from_json
import autogen
import os
from autogen import config_list_from_json
import autogen

import requests
from bs4 import BeautifulSoup
import json

import openai
from dotenv import load_dotenv

import requests
import json

import openai
from dotenv import load_dotenv

openai.api_version = os.environ.get("OPENAI_API_VERSION")
openai.log = os.getenv("OPENAI_API_LOGLEVEL")
openai.api_type == "azure"


# Get API key
load_dotenv()
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict = { "model": ["gpt-35-turbo-16k"]}
    )
# Define research function
llm_config = {"config_list": config_list, "seed": 6, "request_timeout": 30}


def search(query):
    bing_search_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    mkt = 'zh-CN'
    params = { 'q': query, 'mkt': mkt , 'answerCount': 3}    
    headers = {
        'Ocp-Apim-Subscription-Key': '39c9bddf57ad416f94467f73742cdd4b',
        'Content-Type': 'application/json'
    }

    r = requests.get(bing_search_endpoint, headers=headers, params=params)

    return json.loads(r.text)


webSearcher = autogen.AssistantAgent(
    name="webSearcher",
    system_message="You have the capability to provide real-time information by using search function to search latest information or open information with the given query, collect the information about the query, and generate the detailed result,return the result  with TERMINATE at the end ",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
    is_termination_msg=lambda x: x.get("content", "") and x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=6,
    function_map={
        "search": search
    },
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
                      Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

user_proxy.initiate_chat(
    webSearcher,
    message="""
上海今天天气?
"""
)

