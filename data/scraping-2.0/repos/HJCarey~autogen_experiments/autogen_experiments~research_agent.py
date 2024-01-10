import os
import argparse
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

ERROR_KEYWORDS = ["Access denied", "Access Denied", "403",
                  "access denied", "Unauthorized", "unauthorized"]

# Get API key
load_dotenv()
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
openai.api_key = os.getenv("OPENAI_API_KEY")

def search(query):
    """
    This function makes a POST request to Google's search API
    to search for the user input 'query'.

    Params:
    query : str
    
    Returns:
    response : requests.models.Response
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
    })
    headers = {
        "X-API-KEY": "1ccb743eaad7f699de4603b108dcfdff9c6bb4f5",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()


def scrape(url: str):
    """
    Scrapes a website, returning the textual content of the site.
    This will also summarize the content if it is too large of a 
    website.

    Params:
    url : str
        The URL of the website to scrape.

    Returns:
    text/output : str
        The textual content of the website.
    """
    print("Scraping website: ", url)

    # Define headers and data for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    data = {
        "url": url,
    }

    # Convert the data to JSON
    data_json = json.dumps(data)

    # browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
    browserless_api_key = "183e0636-74b3-46dd-b184-e25027b5ec57"

    # Send the POST request
    response = requests.post(
        f"https://chrome.browserless.io/content?token={browserless_api_key}",
        headers=headers,
        data=data_json,
    )

    # Format our data using BeautifulSoup
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        print("Scraped text: ", text)
        if any(keyword in text for keyword in ERROR_KEYWORDS):
            return f"{url} : Not a scrapable website."
        if len(text) > 8000:
            output = summary(text)
            return output
        else:
            return text
        
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(content):
    """
    This function summarizes the content of a website using Langchain and OpenAI.
    It takes the content of a website and asks GPT to summarize it.

    Params:
    content : str
        The content of the website to summarize.

    Returns:
    output : str
        The summary of the content.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for research purposes:
    {text}
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs,)

    return output


def research(query):
    """
    This function defines our 'Researcher' AI agent that uses the functions
    defined above to reach out to the internet and search for information about the
    incoming query.

    Params:
    query : str
        The query to search for.
    """
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "Google search for relevant content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scrape a website based on url.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of website to scrape.",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list
    }

    researcher = autogen.AssistantAgent(
        name="researcher",
        system_message="""Research about a given query, collect as much information as
        possible, and generate detailed research results with lots of technical details
        with all reference links attached. Add 'TERMINATE' to the end of the report.
        """,
        llm_config=llm_config_researcher,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
        "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": search,
            "scrape": scrape,
        }
     )

    user_proxy.initiate_chat(researcher, message=query)

    # Set the receiver to be the researcher, and get a summary of the research report
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
        """Give me the research report that you just generated again, return ONLY the 
        report and reference links""",
        researcher,
    )

    # Return the last message the expert received
    return user_proxy.last_message()["content"]


def main(args):
    research_agent_task = args["task"]

    output = research(research_agent_task)

    # Save output query to a file
    with open("output.txt", "w") as f:
        f.write(output)