import os
import openai
import sys
from langchain.prompts import PromptTemplate
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import datetime
from langchain.chat_models import ChatOpenAI

prompt_template = """
[Role]
Act as a Chef Assitant that follow the steps and give a results at the end.

[Steps]

Step 1 : Understand the Given Ingrident
Step 2 : Undrstand the Given Cusine , Mood and Time Span avalabile
Step 3 : According to all the Above Factors Suggest Dishes for the number of people such that least amount of ingrident are used while providing best taste and nutrition value to satify meal for {person_number} person.
Step 4 : Make atleast 5 Dishes suggestion with basic info that would be needed
Step 5 : Use the above information and give that in a Dictionary format so it is easy to distinguish and understand.

[Steps End]

Ingrident List : {ingrident_list}
Motivation : {motivation}
"""

def get_response(ingrident,motivation):
  chat = ChatOpenAI(model="gpt-3.5-turbo")
  prompt = PromptTemplate.from_template(prompt)
  prompt.format(ingrident_list=ingrident, motivation=motivation)
  response = chat.complete(prompt)
  return response

response = get_response()
print(response)

