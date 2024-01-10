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
from langchain import PromptTemplate
import openai
from dotenv import load_dotenv

load_dotenv()
config_list = config_list_from_json(env_or_file='OAI_CONFIG.json')
openai.api_key = os.getenv("OPENAI_API_KEY")

def search(query):
    pass
def scrape(url: str):
    pass
def summary(content):
    pass
def wiret_content(research_material, topic):
    pass


