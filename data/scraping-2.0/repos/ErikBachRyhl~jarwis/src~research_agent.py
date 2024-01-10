import os, json, requests
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup

import streamlit as st
from langchain.schema import SystemMessage

# Wikipedia
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

# Arxiv
from langchain.utilities import ArxivAPIWrapper

# Creating files
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit

# Own scripts
import node_depths

# Temporary directory for llm
working_directory = "/Users/erik/Documents/Obsidian/research"
file_management_toolkit = FileManagementToolkit(root_dir=working_directory)

tools = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

read_tool, write_tool, list_tool = tools

# load api keys
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Initialize wikipedia and arxiv
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivAPIWrapper()

# Summary function (map reduce)
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """

    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

# Wikipedia Tool
def wikipedia_search(query):
    response = wikipedia.run(query)
    return response

# ArXiv Tool
def arxiv_search(query):
    response = arxiv.run(query)
    return response

# Serp search Tool
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# Scraping Tool
def scrape_website(objective: str, url: str):
    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
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
        print("Content: ", text)

        # Use map-reduce summary if the content text is long
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    
    def _arun(self, url: str):
        raise NotImplementedError("Error here")

# Langchain agent with the tools above
tools = [
    Tool(
        name = "wikipedia_article",
        func = wikipedia_search,
        description="Only use this if the research objective matches the name of a wikipedia article. If there is no good candidate article, resort to the other tools."
    ),
    Tool(
        name = "search", 
        func = search, 
        description = "Useful for when you need to answer questions about current events or data that wikipedia couldn't help with. You should ask targeted questions."
    ),
    ScrapeWebsiteTool(),
    Tool(
        name = "arxiv_paper",
        func = arxiv_search,
        description="Useful for researching the latest advances in a research field. Only use this tool if the user specifies that they want the latest advances."
    )
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ Write a thorough, fact-based explanation on the topic in the style of Richard Feynman.
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
memory = ConversationBufferMemory(memory_key="memory", return_messages=True, llm = llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research Topic")

#     if query:
#         st.write("Doing research on ", query)

#         result = agent({"input": query})

#         st.info(result['output'])

def main(topic):
    print("Doing research on ", topic)

    result = agent({"input": topic})

    resulting_text = result['output']

    # Append-adds at last
    current_file = open(f"{working_directory}/{topic}.md", "a")  # append mode
    current_file.write(f"\n\n\n\nAgent's research results:\n\n{resulting_text}")

test_topic = "Quantum Mechanics"

to_be_researched, skipped = node_depths.get_topic_list(working_directory, test_topic, 1)

for topic in to_be_researched:
    main(topic)