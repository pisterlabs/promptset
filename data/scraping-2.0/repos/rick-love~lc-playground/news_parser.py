import datetime
from config import get_OpenAI, get_NewsAPIKey
from openai import OpenAI
from newsapi import NewsApiClient
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Import Pydantic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator, validator
from typing import List


# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# Set the API key for NewsAPI
try:
    newsapi = NewsApiClient(api_key=get_NewsAPIKey())
except Exception as e:
    raise Exception(f"Error setting API key for NewsAPI: {e}")

# LLMS
llm_model = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0, model=llm_model)

# Get Dates
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Get News from NewsAPI
sources = newsapi.get_sources(country='de', language='de') #'wired-de', 'techcrunch-de', 't3n', 'spiegel-online', 'reuters', 'gruenderszene', 'focus', 'der-tagesspiegel', 'der-standard', 'der-spiegel', 'die-zeit', 'google-news-de', 'handelsblatt', 'heise', 'n-tv', 'techcrunch', 'the-verge', 'wired'
all_articles = newsapi.get_everything(sources='bbc-news', page=1, from_param=yesterday, to=today)#, language='de', sort_by='relevancy')
# print(sources)
print(all_articles)
# top_headlines = newsapi.get_top_headlines(country='de', category='science', language='de')

# first_article = top_headlines['articles'][0]



# Create Data class
# class Article(BaseModel):
#     source: object = Field(description="contains an 'id' and 'name' of where the news is from")
#     author: str = Field(description="author of the news article")
#     title: str = Field(description="title of the news article")
#     description: str = Field(description="description of the news article")
#     url: str = Field(description="url of the news article")
#     publishedAt: str = Field(description="date and time the news article was published")
#     content: str = Field(description="content of the news article")
    
#     # Validation
#     @field_validator('content')
#     def has_content(cls, v):
#         if v != None:
#             raise ValueError("Content must exist")
#         return v

# # Set up the parser and validation for the output
# pydantic_parser = PydanticOutputParser(pydantic_object=Article)
# format_instructions = pydantic_parser.get_format_instructions()

# # Create the prompt
# news_template_revised = """
# From the following news article, please extract the following information:

# news: {first_article}
# {format_instructions}
# """
# print(first_article)
# print(top_headlines.keys())
# print(top_headlines['articles'][0].keys())

# prompt = ChatPromptTemplate.from_template(template=news_template_revised)
# news_articles = prompt.format_messages(news=top_headlines, format_instructions=format_instructions)

# # Get the response
# response = chat(news_articles)

# print(response.content)

