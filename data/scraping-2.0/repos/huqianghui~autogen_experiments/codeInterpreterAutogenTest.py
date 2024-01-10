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



assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "work_dir",
        "use_docker": False,
    },
)

if __name__ == "__main__":
    user_proxy.initiate_chat(assistant, message="according the excel file:/Users/huqianghui/Downloads/Azure_resource.xlsx, 通过location维度，统计resourceId，用bar图展示")