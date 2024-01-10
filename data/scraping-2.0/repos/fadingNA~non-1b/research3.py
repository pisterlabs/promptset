import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen

# Load the .env file
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST.json")


def web_search(input):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": input,
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(f"Response: {response.text}")
    return response.text


def summary_text(content, task):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([content])
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
    map_prompt = """
    Write a summary the following text for {task}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        map_prompt, input_variables=["text", "task"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    output = summary_chain.run(input_documents=documents, objective=task)

    return output


def web_scraping(objective: str, url: str):
    print("Web Scraping...")
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url,
    }

    data_json = json.dumps(data)
    response = requests.post(
        f"https://chrome.browserless.io/content?token={browserless_api_key}",
        headers=headers,
        data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        print("Web Scraping Done! : ", text)
        if len(text) > 10000:
            output = summary_text(text, objective)
            return output
        else:
            return text
    else:
        print(f"Error: {response.status_code}")


def get_airtable_records(base_id, table_id):
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {
        'Authorization': f"Bearer {airtable_api_key}",
    }
    response = requests.request("GET", url, headers=headers)
    return response.json()


def update_single_airtable_record(base_id, table_id, id, fields):
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
        "Content-Type": "application/json"
    }

    data = {
        "records": [{
            "id": id,
            "fields": fields
        }]
    }

    response = requests.patch(url, headers=headers, data=json.dumps(data))
    data = response.json()
    return data




user_proxy = UserProxyAgent(name="user_proxy",
                            is_termination_msg=lambda msg: "Terminate" in msg["content"],
                            human_input_mode="ALWAYS",
                            max_consecutive_auto_reply=1)

researcher = GPTAssistantAgent(name="researcher",
                               llm_config={
                                   "config_list": config_list,
                                   "assistant_id": "asst_31CUYLlH6hCsvkN87p7jSBzg",
                               })

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "web_search": web_search
    }
)

research_manager = GPTAssistantAgent(name="research_manager",
                                     llm_config={
                                         "config_list": config_list,
                                         "assistant_id": "asst_xLx5q2KxPnOkXhS0ChVPMrsG",
                                     })


research_director = GPTAssistantAgent(name="research_director",
                                      llm_config={
                                          "config_list": config_list,
                                          "assistant_id": "asst_GrQIIfBToByaL7IOeYcQMwib",
                                      })
research_director.register_function(
    function_map={
        "get_airtable_records": get_airtable_records,
        "update_single_airtable_record": update_single_airtable_record,
    }
)

group_chat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, research_director],
                               messages=[], max_round=10)
group_chat_manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list})


message = """
Research the api weather data which one is forecast or historical data
provide. focus on the hourly data.
list: https://airtable.com/appqSQcliGnkSLbaa/tbl74uZ3blk6CEwAE/viwivoHsMjvlq0AB8?blocks=hide
"""
user_proxy.initiate_chat(group_chat_manager, message=message)