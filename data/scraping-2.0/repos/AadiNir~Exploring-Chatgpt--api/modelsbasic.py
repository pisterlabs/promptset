import os
from langchain.llms import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
a = os.environ['openai_apikey']


llm = OpenAI(openai_api_key=a,temperature=0.9)

llm.predict("What would be a good company name for a company that makes colorful socks?")
