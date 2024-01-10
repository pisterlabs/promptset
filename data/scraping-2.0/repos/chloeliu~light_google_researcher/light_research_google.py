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
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import re

from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": 10

    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    parsed_dict = json.loads(response.text)
    organic_results = parsed_dict.get("organic", [])
    # print(response.text)
    return organic_results


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
    ),
]
            # 2/ If there are url of relevant links & articles,you will scrape it to gather more information
            # 3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ You should do a different search with different search query if more research is needed
            3/ You should not make things up, you should only write facts & data that you have gathered
            4/ you should always refernece the source of the information you gathered

            output the report in the following format:
            ```
            TLDR: (answer to user request precisely)
            Key points: (analysis from information gathered in short precise description (make sure you link to reference urls) )
            Further reading : (full list of relevant links and the snippet or summarization of the articles) 
            ``` 
            output:
            """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=10,
)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI Reseacher")

    st.header("Find answer faster with AI")
    query = st.text_input("Search for:")
    if query:
        st.write("Doing research for ", query)
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        result = agent({"input": query}, callbacks=[st_cb])

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
# app = FastAPI()


# class Query(BaseModel):
#     query: str


# @app.post("/")
# def researchAgent(query: Query):
#     query = query.query
#     content = agent({"input": query})
#     actual_content = content['output']
#     return actual_content
