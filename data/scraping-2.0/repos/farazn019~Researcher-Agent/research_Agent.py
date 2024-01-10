import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

import streamlit as st

load_dotenv()

duckduckgo_search = DuckDuckGoSearchResults()


Headers = {
    'user-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
}

def parse_html_content(input_content) -> str:
    html_output = BeautifulSoup(input_content, 'html.parser')
    html_content = html_output.get_text()

    return html_content

def get_website_page(input_url) -> str:
    response = requests.get(input_url, headers=Headers)
    return parse_html_content(response.content)


web_fetcher = Tool.from_function (
    func=get_website_page,
    name="WebFetcher",
    description="Retrieves the content of a web page"
)


template_prompt = "Summarize this content: {content}"
large_language_model = ChatOpenAI(model="gpt-3.5-turbo-16k")

large_language_model_chain = LLMChain(
    llm = large_language_model,
    prompt=PromptTemplate.from_template(template_prompt)
)

summarize_tool = Tool.from_function(
    func = large_language_model_chain.run,
    name = "Summarization Tool",
    description = "This summarizes the contents of a website"
)


tools = [duckduckgo_search, web_fetcher, summarize_tool]

research_agent = initialize_agent(
    tools = tools,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm = large_language_model,
    verbose = True
)

#prompt = "Research how to grow out a business using newsletters. Use your tools to search and summarize content into a guide on how to use the newsletters to effectively grow a business."
#prompt = open()
with open("sample_prompt.txt", "r") as prompt:
    prompt_input = prompt.read()
    #print(prompt_input)
    print(research_agent.run(prompt_input))

#prompt_input_text = st.text_input('Enter Prompt', 'Please enter your prompt')
#st.write('The current title is, ' + prompt_input_text)

#print(research_agent.run(prompt_input_text))

