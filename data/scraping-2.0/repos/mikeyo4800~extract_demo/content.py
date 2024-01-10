import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def find_internal_links(base_url):
    internal_links = set()
    # Fetch the content of the base URL
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to retrieve {base_url}")
        return internal_links

    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links in the page
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.attrs['href']
        # Join the relative URL with the base URL
        full_url = urljoin(base_url, href)
        # Check if the URL belongs to the same domain
        if full_url.startswith(base_url):
            internal_links.add(full_url)

    joined_links = "\n".join(internal_links)  # Join the links with newline as the separator


    return joined_links

# 1. Tool for search


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


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
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
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


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
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the user"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(name="find_internal_links", func=find_internal_links, description="useful to find all internal links of a website before scraping websites"),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""Transform into SEOCONTENTMASTER, an AI writing expert with vast experience in writing techniques and frameworks. I collect and analyze data from your entire website to create a unique, engaging, and informative article that will rank high on search engines.
    I will use a conversational style, employing informal tone, personal pronouns, active voice, rhetorical questions, and analogies and metaphors to engage the reader. The headings will be bolded and formatted using Markdown language, with appropriate H1, H2, H3, and H4 tags
    The final piece will be a 2000-word article, featuring a conclusion paragraph and five unique FAQs after the conclusion. My approach will ensure high levels of perplexity and burstiness without sacrificing context or specificity."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

def create_fb_content(niche="general-contractors", company="ABC", website="https://www.abc.com/", topic="roofing", title="Roofing Services"):
    
    query = f"Write SEO content for this {company} who are {niche}. They want content about their {topic} services with this {title}. Use their website, {website}, for reference."
    
    result = agent({"input": query})
    
    return result["output"]