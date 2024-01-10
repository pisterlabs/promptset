import os
import requests
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, config_list_from_json
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI

import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface 
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas as persona
import memgpt.humans as human
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss
import openai

load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

#Functions
#Google search function
def google_search(search_keyword):    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("RESPONSE:", response.text)
    return response.text

#summary function
def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-1106")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
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
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

#web scraping function
def web_scraping(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

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
    response = requests.post(f"https://chrome.browserless.io/content?token={browserless_api_key}", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")        


user_proxy = UserProxyAgent(
   name="User_Proxy",
   system_message="A user proxy. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
#    code_execution_config=False,
#    human_input_mode="Always",
#    max_consecutive_auto_reply=1,
#    is_termination_msg=lambda msg: "TERMINATE" in msg["content"]
) 

# This is how memgpt talks to autogen
interface = autogen_interface.AutoGenInterface()
persistence_manager = InMemoryStateManager()
persona = '''I am a world class reseacher who can do detailed research on any topic and produce fact based results. I do not make things up. I will try as hard as possible to gather facts and data to back up the research. 
    I will make sure I complete the objective above with the following rules:
    1. I should do enough research to gather as much information as possible about the objective.
    2. If there is a URL of relevant links and articles, I will scrape it to gather more information.
    3. After scraping and searching, I should think to myself, "are there any new things I should search and scrape based on the data I collected to increase research quality?" If the answer is yes, I will continue, but I won't do this more than 3 iterations. 
    4. I will not make things up. I will only write facts and data I have gathered. 
    5. In the final output, I will include all reference data and links to back up my research. I should include all reference data and links to back up my research.
    6. I will not use G2 or Linkedin. They are mostly out dated data. 
'''
human = "I\'m a team manager at a FAANG tech company."
memgpt_agent = presets.use_preset(presets.DEFAULT, 'gpt-3.5-turbo-1106', persona, human, interface, persistence_manager)


#MemGPT researcher
#use create_memgpt_autogen_agent_from_config()
researcher = AssistantAgent(
    name="Researcher",
    agent=memgpt_agent,
)

researcher.register_function(
    function_map={
        "Web_scraping": web_scraping,
        "google_search": google_search
    }
)

research_manager = AssistantAgent(
    name="research_manager",
    system_message='''You are a research manager. You are harsh and relentless. You will first try to generate 2 actions a researcher can take to find the information needed. Try to avoid linkedin, or other gated websites that don't allow scraping. You will review the result from the researcher, and always push back if the researcher didn't find the information. Be persistent. For example, if the researcher does not find the correct information, say, "No, you have to find the information. Try again.", and propose another method to try if the researcher can't find an answer. Only after the researcher has found the information will you say, 'TERMINATE'.
''',
    llm_config={"config_list": config_list},
    
)


groupchat = GroupChat(agents=[user_proxy, researcher, research_manager], messages=[], max_round=10)

# user_proxy.initiate_chat(director, message=message)

manager = GroupChatManager(
    groupchat=groupchat, 
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager,
    message = """What physical therapy exercises should a patient perform for the next 8 weeks one day out from ACL surgery based on the clinical practice guidelines?"""
)

