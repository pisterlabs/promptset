import os
from autogen import config_list_from_json
import autogen

import requests
from bs4 import BeautifulSoup
import json

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import openai
from dotenv import load_dotenv

dotenv_path = '../.env' 
load_dotenv(dotenv_path)
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

openai.api_key = OPEN_AI_API_KEY

config_list_gpt4 = [
    {   
        'model': 'gpt-4',
        'api_key': openai_api_key,
        "request_timeout": 600,
    }
]

config_list_gpt35 = [
    {   
        'model': 'gpt-3.5-turbo-16k',
        'api_key': openai_api_key,
        "request_timeout": 600,
    }
]

# RESEARCH Function (use GPT4)
# search
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

    return response.text

# scrape
def scrape(url: str):
    # scrape url

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
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        if len(text) > 10000:
            output = summary(text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# summarize
def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for a research purpose:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs)

    return output

def research(query):
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "Search the query on Google",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to be searched"
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scrape website based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The url of the website to be scraped"
                        },
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list_gpt4
    }

    researcher = autogen.AssistantAgent(
        name = "researcher",
        system_message = """
        Research about a given query, 
        collect as many information as possible by scraping relevant urls, 
        and generate detailed research essay with all reference links attached; 
        Add TERMINATE to the end of the research reportonce you have scraped sufficient information;
        """,
        llm_config=llm_config_researcher,
    )

    user_proxy = autogen.UserProxyAgent(
        name = "user_proxy",
        code_execution_config = {"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map = {
            "search": search,
            "scrape": scrape,
        }
    )

    user_proxy.initiate_chat(
        researcher,
        message=query,
    )

    # set researcher as message receiver and get summary of the research report
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
       "Give me the research report that just generated again, return ONLY the report & reference links", 
       researcher 
    )

    # return the last message the user_proxy received
    return user_proxy.last_message()["content"]

# result = research("what is microsoft autogen?")

# WRITE CONTENT Function (use GPTX)
def write_content(research_material, topic):
    editor = autogen.AssistantAgent(
        name="editor",
        system_message="""
        You are a senior editor of a blog, 
        you will define the structure of a short blog post based on material 
        provided by the researcher, and give it to the writer to write the blog post.
        """,
        llm_config={"config_list": config_list_gpt4}
    )

    writer = autogen.AssistantAgent(
        name="writer",
        system_message="""
        You are a writer of a blog,
        you will write a short blog post based on the structure defined by the editor, 
        and feedback from reviewer; 
        After 2 rounds of content iteration, add TERMINATE to the end of the message
        """,   
        llm_config={"config_list": config_list_gpt4}
    )     

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="""
        You are a world class news blog content critic, 
        you will review & critic the written blog and provide feedback to writer.
        After 2 rounds of content iteration, add TERMINATE to the end of the message.
        """,
        llm_config={"config_list": config_list_gpt4},
    )

    user_proxy = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with editor to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, editor, writer, reviewer],
        messages=[],
        max_round=20)
    manager = autogen.GroupChatManager(groupchat=groupchat)

    user_proxy.initiate_chat(
        manager, message=f"Write a blog about {topic}, here is the material: {research_material}")

    user_proxy.stop_reply_at_receive(manager)
    user_proxy.send(
        "Give me the blog that just generated again, return ONLY the blog, and add TERMINATE in the end of the message", manager)

    # return the last message the expert received
    return user_proxy.last_message()["content"]

# result = write_content(research("how are anthropic and openAI different?"), "AI systems")

# Define content assistant agent
llm_config_content_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "research about a given topic, return the research material including reference links",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to be researched about",
                        }
                    },
                "required": ["query"],
            },
        },
        {
            "name": "write_content",
            "description": "Write content based on the given research material & topic",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "research material of a given topic, including reference links when available",
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic of the content",
                        }
                    },
                "required": ["research_material", "topic"],
            },
        },
    ],
    "config_list": config_list_gpt4}

content_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    system_message="You are a writing assistant, you can use research function to collect latest information about a given topic, and then use write_content function to write a very well written content; Reply TERMINATE when your task is done",
    llm_config=llm_config_content_assistant,
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="TERMINATE",
    function_map={
        "write_content": write_content,
        "research": research,
    }
)

user_proxy.initiate_chat(
    content_assistant, message="write a blog post about the impact of Ai developments on the legal industry")