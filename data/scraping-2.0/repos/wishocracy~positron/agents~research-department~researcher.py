import os
import yaml
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen

# Load configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load environment variables
load_dotenv()
config_list = config_list_from_json(config["llm_config_list"])


# ------------------ Create functions ------------------ #

# Function for Google search
def google_search(search_keyword):
    url = config["google_search_url"]

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': config["serper_api_key"],
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text


# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


def web_scraping(objective: str, url: str):
    # Scrape website and summarize the content based on objective
    print("Scraping website...")

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url
    }

    data_json = json.dumps(data)

    response = requests.post(f"{config['browserless_url']}?token={config['browserless_api_key']}", headers=headers,
                             data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


# ------------------ Create agent ------------------ #

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
)

researcher = GPTAssistantAgent(
    name="researcher",
    llm_config={
        "config_list": config_list,
        "assistant_id": config["assistant_ids"]["researcher"]
    }
)

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search
    }
)

research_manager = GPTAssistantAgent(
    name="research_manager",
    llm_config={
        "config_list": config_list,
        "assistant_id": config["assistant_ids"]["research_manager"]
    }
)

director = GPTAssistantAgent(
    name="director",
    llm_config={
        "config_list": config_list,
        "assistant_id": config["assistant_ids"]["director"]
    }
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, research_manager, director],
    messages=[],
    max_round=config["group_chat_settings"]["max_round"]
)

group_chat_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# ------------------ Start conversation ------------------ #
init_message = config["init_message"]
user_proxy.initiate_chat(group_chat_manager, message=init_message)
