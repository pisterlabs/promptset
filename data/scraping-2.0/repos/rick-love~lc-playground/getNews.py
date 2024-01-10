from config import get_NewsAPIKey, get_OpenAI
from newsapi import NewsApiClient
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# Set the API key for NewsAPI
try:
    newsapi = NewsApiClient(api_key=get_NewsAPIKey())
except Exception as e:
    raise Exception(f"Error setting API key for NewsAPI: {e}")

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")

model = OpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

# Get the top headlines for a given query
sources = newsapi.get_sources()


# top_headlines = newsapi.get_top_headlines(q='ukraine', country='us', category='business', language='en')
top_headlines = newsapi.get_top_headlines(country='de', language='de')
news_response = """
source: {sources}
headlines: {top_headlines}
"""


# prompt_template = ChatPromptTemplate.from_template(prompt)
# headline_list = prompt_template.format_messages(top_headlines=top_headlines, prompt=prompt)
# response = headline_list
# print(type(response))

# print(type(top_headlines)) # dict
# print(sources)
# print(top_headlines)
