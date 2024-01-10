import autogen
import os
from autogen import config_list_from_json
import autogen

import requests
from bs4 import BeautifulSoup
import json

import openai
from dotenv import load_dotenv

config_list = autogen.config_list_from_json(env_or_file="/Users/huqianghui/Downloads/1.乐歌-openAI/web-scraper/OAI_CONFIG_LIST")

llm_config={
    "request_timeout": 600,
    "seed": 44,                     # for caching and reproducibility
    "config_list": config_list,     # which models to use
    "temperature": 0,               # for sampling
}


def scrape(url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        "https://chrome.browserless.io/content?token=043abdc1-b765-4298-b7b8-b2c39f6ce27e", headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


agent_assistant = autogen.AssistantAgent(
    name="agent_assistant",
    llm_config=llm_config,
    function_map={
        "scrape": scrape
    },
    system_message="You have the capability to scrape internet information or web url by using scrape function to scrape information with the given url, collect the information about the query, and generate the detailed result,return the result. ",
)

agent_proxy = autogen.UserProxyAgent(
    name="agent_proxy",
    human_input_mode="NEVER",           # NEVER, TERMINATE, or ALWAYS   # TERMINATE - human input needed when assistant sends TERMINATE 
    max_consecutive_auto_reply=6,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "agent_output",     # path for file output of program
        "use_docker": False,            # True or image name like "python:3" to use docker image
    }
)

agent_proxy.initiate_chat(
    agent_assistant,
    message="""I need you to write a python script that will do the following:
    1. go to airbnb
    2. search for an Austin Texas stay from Oct 10, 2023 - Oct 11, 2023
    3. gather the results, no more than 10. The class html div to search for is "c4mnd7m dir dir-ltr".
    4. show the results by matlibplot's bar chart
    """,
)