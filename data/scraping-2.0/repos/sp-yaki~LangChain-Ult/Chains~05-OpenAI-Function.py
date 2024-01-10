from typing import Optional

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

llm = ChatOpenAI(model='gpt-3.5-turbo-0613')

class Scientist():
    
    def __init__(self,first_name,last_name):
        self.first_name = first_name
        self.last_name = last_name
        
json_schema = {"title": "Scientist",
               "description": "Information about a famous scientist",
               "type": "object",
               "properties":{
                   "first_name":{'title':'First Name',
                                 'description': "First name of scientist",
                                 "type": "string"},
                   "last_name":{'title':'Last Name',
                                 'description': "Last name of scientist",
                                 "type": "string"},
               },
                "required": ['first_name','last_name']
              }

template = 'Name a famous {country} scientist'
# human_prompt = HumanMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_template(template)

chain = create_structured_output_chain(json_schema,llm,chat_prompt,verbose=True)
result = chain.run(country='American')
print(result)

