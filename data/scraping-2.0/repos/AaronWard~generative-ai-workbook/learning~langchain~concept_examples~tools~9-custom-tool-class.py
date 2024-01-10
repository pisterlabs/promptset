"""
This Agent will query hacker news https://news.ycombinator.com/ to find articles related to
LLM's. This Agent will use a custom tool to strip html into plane text, send that text to an LLM
and provide a summarized version of the story


https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html#subclassing-the-basetool-class
https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81

Langchain agents = Tools + Chains

"""

import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv

from langchain import OpenAI
from langchain import PromptTemplate
from langchain.tools import BaseTool
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import SystemMessagePromptTemplate

load_dotenv(Path('./.env'))
llm = OpenAI(temperature=0)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"


class HackerNewsTool(BaseTool):
    name = "Hacker News Tool"
    description = "Used to extract all titles and web links for articles on the front page of hacker news"
    headers = {"user-agent": USER_AGENT}

    def _run(self, webpage: str):
        response = requests.get(webpage, headers=self.headers)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")

            # Find all the 'a' tags in the HTML, ignoring the lower
            # level span elements
            links = soup.select('span.titleline > a')

            parsed_text = ""
            # Iterate over each link and extract the 'href' attribute and text
            for link in links:
                href = link.get('href')
                text = link.text
                parsed_text += f"{text}\n{href}\n\n"

            return parsed_text

        stripped_content = strip_html_tags(html_content)

        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

# Make make article extraction tool
class ParseArticleTool(BaseTool):
    name = "Extract Article Tool"
    description = "Used to extract text from <p> elements in webpage articles"
    headers = {"user-agent": USER_AGENT}

    def _run(self, webpage: str):
        response = requests.get(webpage, headers=self.headers)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")

            paragraph = soup.select('p')

            parsed_text = ""
            # Iterate over each link and extract the 'href' attribute and text
            for par in paragraph:
                text = par.text
                parsed_text += f"{text}\n"
            
            return parsed_text
        
        stripped_content = strip_html_tags(html_content)

        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

# Declare Tools
page_getter = HackerNewsTool()
parse_text_getter = ParseArticleTool()
tools = [page_getter]

# Create memory

# Prompts
hacker_news_prompt = "You need pull all titles and links from {webpage} using HackerNewsTool"

filter_prompt = \
"""You need to find all article titles relating to AI and LLM's from {input}
Exclude anything not relating to artificial intelligence. Here is an example:

Observation:
```
Glass or plastic: which is better for the environment?
https://www.bbc.com/future/article/20230427-glass-or-plastic-which-is-better-for-the-environment

Comparing Physician and AI Responses to Patient Questions Posted to Public Forum
https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2804309

ChatGPT for Google – Display ChatGPT Response Alongside Search Engine Results
https://chrome.google.com/webstore/detail/chatgpt-for-google/jgjaeacdkonaoafenlfkkkmbaopkbilf
```

This is what your response should look like:
```
Comparing Physician and AI Responses to Patient Questions Posted to Public Forum
https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2804309

ChatGPT for Google – Display ChatGPT Response Alongside Search Engine Results
https://chrome.google.com/webstore/detail/chatgpt-for-google/jgjaeacdkonaoafenlfkkkmbaopkbilf
```
"""

# Initialize agents
extraction_agent = initialize_agent(
    agent='chat-zero-shot-react-description',
    tools=[page_getter],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    # memory=memory
)
# extraction_agent.agent.llm_chain.prompt.messages[0].prompt.template = hacker_news_prompt
# extraction_agent.agent.llm_chain.prompt.template = hacker_news_prompt


filtering_agent = initialize_agent(
    agent='chat-zero-shot-react-description',
    tools=[page_getter],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    # memory=memory
)
# filtering_agent.agent.llm_chain.prompt.messages[0].prompt.template = filter_prompt
# filtering_agent.agent.llm_chain.prompt.template = filter_prompt


# Run agent
result = extraction_agent("https://news.ycombinator.com/newest")
result = filtering_agent(result)

# print results
print(result)

# ========================================================================================
'''
# Create memory
# memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=3,
#     return_messages=True
# )

# Initialize agent
# conversational_agent = initialize_agent(
#     agent='conversational-react-description', 
#     tools=tools, 
#     llm=llm,
#     verbose=True,
#     max_iterations=4,
#     # memory=memory
# )

# Prompt template
# extract_prompt = \
# """You need to find all article titles relating to AI and LLM's from {webpage}
# Exclude anything not relating to artificial intelligence. Here is an example:

# Observation:
# ```
# Glass or plastic: which is better for the environment?
# https://www.bbc.com/future/article/20230427-glass-or-plastic-which-is-better-for-the-environment

# Comparing Physician and AI Responses to Patient Questions Posted to Public Forum
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2804309

# ChatGPT for Google – Display ChatGPT Response Alongside Search Engine Results
# https://chrome.google.com/webstore/detail/chatgpt-for-google/jgjaeacdkonaoafenlfkkkmbaopkbilf
# ```

# This is what your response should look like:
# ```
# Comparing Physician and AI Responses to Patient Questions Posted to Public Forum
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2804309

# ChatGPT for Google – Display ChatGPT Response Alongside Search Engine Results
# https://chrome.google.com/webstore/detail/chatgpt-for-google/jgjaeacdkonaoafenlfkkkmbaopkbilf
# ```
# """

# extract_prompt_template = PromptTemplate(
#     input_variables=['webpage'],
#     template=extract_prompt
# )

# conversational_agent.agent.llm_chain.prompt.template = extract_prompt


# # Run agent
# conversational_agent("https://news.ycombinator.com/newest")



# print results
'''

# ============