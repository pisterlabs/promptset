import requests
from newspaper import Article
from dotenv import dotenv_values

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPEN_AI_API)
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPEN_AI_API)

article_url = "https://www.howzat.com/football/football-rules-and-regulations.html"
session = requests.Session()

try:
    response = session.get(article_url)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(article.text)

    else:
        print("Error: Invalid URL")

except Exception as e:
    print(f"Error: {e}")

template = """
You have the task of summarizing an article in the following format:

Here's the article you need to summarize:

==================
Title: {article_title}

{article_text}

==================

{output_format_instructions}

"""

# creating a Pydantic model to parse the output
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Summary of the article")

    @validator('summary', allow_reuse=True)
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# creating a parser to parse the output using the Pydantic model
parser = PydanticOutputParser(pydantic_object=ArticleSummary)  

# creating a prompt template with inputs such as template,  article title and text, and the output parser
prompt = PromptTemplate(template=template, 
                        input_variables=['article_title', 'article_text'],
                        partial_variables={"output_format_instructions": parser.get_format_instructions()},  # used to format the output
)

# providing the article title and text to the prompt template to get the formatted prompt
formatted_prompt = prompt.format_prompt(article_title=article.title, article_text=article.text)


messages = [HumanMessage(content=formatted_prompt.to_string())]

response = chat(messages)
print(response.content)


# Use the model to generate a summary
# output = model(formatted_prompt.to_string())

# # Parse the output into the Pydantic model
# parsed_output = parser.parse(output)
# print(parsed_output)