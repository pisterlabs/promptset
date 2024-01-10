import json
from typing import Type

import requests
from bs4 import BeautifulSoup
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from agent import browserless_api_key, llm


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = ("useful when you need to get data from a website url, passing both url and objective to the "
                   "function; DO NOT make up any url, the url should only be from the search results")
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


def scrape_website(objective: str, url: str):
    print("Scraping website...")
    response = browserless_post(url)
    if response.status_code == 200:
        return summarise_response(objective, response)
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summarise_response(objective, response):
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    print("CONTENT:", text)
    if len(text) > 10000:
        output = summary(objective, text)
        return output
    else:
        return text


def browserless_post(url):
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=browserless_headers(), data=browserless_data(url))
    return response


def browserless_data(url):
    return json.dumps({
        "url": url
    })


def browserless_headers():
    return {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }


def summary(objective, content):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    prompt = prompt_template()
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=prompt,
        combine_prompt=prompt,
        verbose=True
    )
    return summary_chain.run(input_documents=text_splitter.create_documents([content]), objective=objective)


def prompt_template():
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    return PromptTemplate(
        template=map_prompt,
        input_variables=["text", "objective"]
    )
